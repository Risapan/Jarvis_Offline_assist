"""Speech-to-text with Whisper on Apple Silicon (MLX)."""

from __future__ import annotations

import re

import mlx_whisper
import numpy as np

# Whisper tends to invent these on silence / background noise.
_HALLUCINATIONS = {
    "", "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", "subtitles by the amara.org community", ".",
}


class Transcriber:
    def __init__(self, model: str, language: str = "en"):
        self.model = model
        self.language = language

    def warmup(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=self.language,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        text = (result.get("text") or "").strip()
        normalized = re.sub(r"[^\w\s.]", "", text.lower()).strip(" .")
        return "" if normalized in _HALLUCINATIONS else text
