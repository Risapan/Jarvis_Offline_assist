"""Central configuration. Every value can be overridden with an env var (JARVIS_<NAME>)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    raw = os.environ.get(f"JARVIS_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    return type(default)(raw)


@dataclass
class Config:
    # Brain
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.6
    llm_ctx: int = 8192  # room for tool schemas + memories + history
    llm_think: bool = False  # qwen3 "thinking" adds seconds of latency; keep off for chat
    history_turns: int = 10  # short-term memory: last N user/assistant pairs

    # Hands (Mac control)
    tools_enabled: bool = True
    confirm_risky: bool = True  # ask "yes/no" before raw shell / AppleScript
    max_tool_rounds: int = 5

    # Ears
    stt_model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    vad_aggressiveness: int = 2  # 0..3, higher = stricter about what counts as speech
    end_silence_ms: int = 900  # silence that ends an utterance
    max_utterance_s: float = 30.0

    # Voice
    tts_model: Path = ROOT / "models" / "kokoro-v1.0.onnx"
    tts_voices: Path = ROOT / "models" / "voices-v1.0.bin"
    voice: str = "bm_george"  # British male; try bm_lewis, bm_daniel, bf_emma
    voice_lang: str = "en-gb"
    voice_speed: float = 1.05

    # Memory
    embed_model: str = "nomic-embed-text"
    db_path: Path = ROOT / "data" / "memory.db"
    recall_k: int = 5
    recall_min_score: float = 0.55
    learn_facts: bool = True

    user_name: str = field(default_factory=lambda: os.environ.get("USER", "sir"))

    def __post_init__(self):
        for f in fields(self):
            current = getattr(self, f.name)
            if isinstance(current, Path):
                setattr(self, f.name, Path(_env(f.name, str(current))))
            else:
                setattr(self, f.name, _env(f.name, current))
