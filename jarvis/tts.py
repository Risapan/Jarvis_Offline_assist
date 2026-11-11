"""Text-to-speech with Kokoro, pipelined so the next sentence is synthesized
while the current one is playing."""

from __future__ import annotations

import queue
import re
import threading

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

_STOP = object()


def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"[*_`#>|~]+", "", text)  # markdown
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)  # bullets
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    def __init__(self, model_path, voices_path, voice: str, lang: str, speed: float):
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.voice, self.lang, self.speed = voice, lang, speed
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue(maxsize=4)
        self._pending = 0
        self._idle = threading.Condition()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def warmup(self) -> None:
        self.kokoro.create("Ready.", voice=self.voice, speed=self.speed, lang=self.lang)

    def say(self, text: str) -> None:
        """Queue text to be spoken. Returns immediately."""
        text = clean_for_speech(text)
        if not text:
            return
        with self._idle:
            self._pending += 1
        self._text_q.put(text)

    def wait(self) -> None:
        """Block until everything queued has finished playing."""
        with self._idle:
            self._idle.wait_for(lambda: self._pending == 0)

    def _synth_loop(self) -> None:
        while True:
            text = self._text_q.get()
            try:
                samples, sr = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
            except Exception as e:  # never let one bad sentence kill the voice
                print(f"\n[tts error] {e}")
                samples, sr = np.zeros(0, dtype=np.float32), 24_000
            self._audio_q.put((samples, sr))

    def _play_loop(self) -> None:
        while True:
            samples, sr = self._audio_q.get()
            if len(samples):
                sd.play(samples, sr)
                sd.wait()
            with self._idle:
                self._pending -= 1
                self._idle.notify_all()
