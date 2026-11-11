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


def _local_path(model: str) -> str:
    """Use the cached copy if the model was already downloaded, so startup never
    touches the network. Falls back to the repo id (downloads on first run)."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:
        return model


class Transcriber:
    def __init__(self, model: str, language: str = "en", prompt: str | None = None):
        self.model = _local_path(model)
        self.language = language
        self.prompt = prompt  # vocabulary hint, e.g. installed app names

    def warmup(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=self.language,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt=self.prompt,
        )
        text = (result.get("text") or "").strip()
        # Drop low-confidence output: Whisper turns noise into fluent-looking gibberish.
        segs = result.get("segments") or []
        if segs:
            avg_logprob = sum(s["avg_logprob"] for s in segs) / len(segs)
            if avg_logprob < -1.0 or max(s["compression_ratio"] for s in segs) > 2.4:
                return ""
        if text and sum(not c.isascii() for c in text) / len(text) > 0.1:
            return ""  # English-only mode: lots of foreign script means garbage
        normalized = re.sub(r"[^\w\s.]", "", text.lower()).strip(" .")
        return "" if normalized in _HALLUCINATIONS else text
