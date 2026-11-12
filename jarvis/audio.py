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
MIN_RMS = 250.0  # absolute minimum, int16 units
GATE_RATIO = 2.5  # speech must be this many times louder than the noise floor


class Listener:
    def __init__(self, aggressiveness: int = 2, end_silence_ms: int = 700, max_utterance_s: float = 30.0):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.end_silence_frames = end_silence_ms // FRAME_MS
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.noise_floor = MIN_RMS / GATE_RATIO
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
        silence = 0

        while True:
            frame = self._frames.get()
            if len(frame) != FRAME_SAMPLES * 2:
                continue
            rms = float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2)))
            loud = rms >= max(MIN_RMS, self.noise_floor * GATE_RATIO)
            is_speech = loud and self.vad.is_speech(frame, SAMPLE_RATE)
            if not is_speech:  # track background level (rises fast, falls slowly)
                a = 0.05 if rms > self.noise_floor else 0.01
                self.noise_floor += a * (rms - self.noise_floor)

            if not speaking:
                pre_roll.append(frame)
                window.append(is_speech)
                if len(window) == window.maxlen and sum(window) >= START_RATIO * window.maxlen:
                    speaking = True
                    voiced.extend(pre_roll)
                    silence = 0
                continue

            voiced.append(frame)
            silence = 0 if is_speech else silence + 1
            if silence >= self.end_silence_frames or len(voiced) >= self.max_frames:
                speech_frames = len(voiced) - silence
                if speech_frames * FRAME_MS < MIN_SPEECH_MS:
                    # too short to be a real request; reset and keep listening
                    voiced.clear()
                    window.clear()
                    pre_roll.clear()
                    speaking = False
                    continue
                pcm = np.frombuffer(b"".join(voiced), dtype=np.int16)
                return pcm.astype(np.float32) / 32768.0
