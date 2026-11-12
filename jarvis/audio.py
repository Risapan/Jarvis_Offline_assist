"""Microphone capture with voice-activity detection.

Audio is read continuously in 30 ms frames. An utterance starts once enough
frames look like speech and ends after `end_silence_ms` of silence. A short
pre-roll buffer keeps the first syllable from being clipped.
"""

from __future__ import annotations

import collections
import queue

import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
PRE_ROLL_MS = 300
START_WINDOW = 10  # frames inspected when deciding speech has started
START_RATIO = 0.6  # fraction of START_WINDOW that must be voiced
MIN_SPEECH_MS = 250  # utterances shorter than this are dropped as noise
# Loudness gate: a frame only counts as speech if it is clearly louder than the room.
# The noise floor adapts to background chatter/TV, so only the close (loud) voice triggers.
MIN_RMS = 200.0  # absolute minimum, int16 units
GATE_RATIO = 1.6  # speech must be this many times louder than the noise floor
FLOOR_WINDOW = 100  # frames (3 s) of history used to estimate the room's background level
FLOOR_PERCENTILE = 30  # quiet end of that window: steady background, not anyone's speech peaks
# End of utterance is judged against the user's OWN voice level, so background talk that is
# quieter than them counts as silence instead of keeping the recording open.
VOICE_DROP = 0.4  # frames below 40% (-8 dB) of the user's recent peak level are "quiet"
END_QUIET_RATIO = 0.8  # utterance ends when this share of the last end_silence_ms is quiet


class Listener:
    def __init__(self, aggressiveness: int = 2, end_silence_ms: int = 700, max_utterance_s: float = 30.0):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.end_silence_frames = end_silence_ms // FRAME_MS
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.noise_floor = MIN_RMS / GATE_RATIO
        self._levels: collections.deque[float] = collections.deque(maxlen=FLOOR_WINDOW)
        self._frames: queue.Queue[bytes] = queue.Queue()
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="int16",
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status):
        self._frames.put(bytes(indata))

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def flush(self) -> None:
        """Drop buffered audio, e.g. Jarvis's own voice picked up while speaking."""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def listen(self) -> np.ndarray:
        """Block until one utterance is captured; return float32 mono audio at 16 kHz."""
        pre_roll: collections.deque[bytes] = collections.deque(maxlen=PRE_ROLL_MS // FRAME_MS)
        window: collections.deque[bool] = collections.deque(maxlen=START_WINDOW)
        voiced: list[bytes] = []
        speaking = False
        voice_level = 0.0  # decaying peak of the user's speech loudness
        recent_quiet: collections.deque[bool] = collections.deque(maxlen=self.end_silence_frames)
        pre_levels: collections.deque[float] = collections.deque(maxlen=PRE_ROLL_MS // FRAME_MS)

        while True:
            frame = self._frames.get()
            if len(frame) != FRAME_SAMPLES * 2:
                continue
            rms = float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2)))
            loud = rms >= max(MIN_RMS, self.noise_floor * GATE_RATIO)
            is_speech = loud and self.vad.is_speech(frame, SAMPLE_RATE)
            if not speaking:  # learn the room only between utterances, never from the user's voice
                self._levels.append(rms)
                if len(self._levels) >= 10:
                    self.noise_floor = float(np.percentile(self._levels, FLOOR_PERCENTILE))

            if not speaking:
                pre_roll.append(frame)
                pre_levels.append(rms)
                window.append(is_speech)
                if len(window) == window.maxlen and sum(window) >= START_RATIO * window.maxlen:
                    speaking = True
                    voiced.extend(pre_roll)
                    voice_level = max(pre_levels)
                    recent_quiet.clear()
                continue

            voiced.append(frame)
            if is_speech:
                voice_level = max(rms, voice_level * 0.997)  # follows the user, decays ~1 dB/s
            quiet = not is_speech or rms < voice_level * VOICE_DROP
            recent_quiet.append(quiet)
            ended = (
                len(recent_quiet) == recent_quiet.maxlen
                and sum(recent_quiet) >= END_QUIET_RATIO * recent_quiet.maxlen
            )
            if ended or len(voiced) >= self.max_frames:
                tail = len(recent_quiet) if ended else 0
                speech_frames = len(voiced) - tail
                if speech_frames * FRAME_MS < MIN_SPEECH_MS:
                    # too short to be a real request; reset and keep listening
                    voiced.clear()
                    window.clear()
                    pre_roll.clear()
                    speaking = False
                    continue
                keep = voiced[: len(voiced) - tail + 6]  # drop the quiet tail (keep ~180 ms)
                pcm = np.frombuffer(b"".join(keep), dtype=np.int16)
                return pcm.astype(np.float32) / 32768.0
