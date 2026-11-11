# Jarvis — Offline Assistant

A real-time voice assistant that runs entirely on your Mac. No cloud and no API keys.

```
Mic ─► VAD ─► Whisper (MLX) ─► Ollama LLM ─► Kokoro TTS ─► Speaker
                                  ▲   │
                                  │   ▼
                            Memory (SQLite + embeddings)
```

| Part | Tech | Latency (M-series, warm) |
|---|---|---|
| Ears | `webrtcvad` + `mlx-whisper` large-v3-turbo | ~0.35 s |
| Brain | Ollama `qwen3:8b`, thinking off, streamed | ~0.5–1 s to first sentence |
| Voice | Kokoro-82M (ONNX), British voice `bm_george` | ~0.4 s per sentence |
| Memory | SQLite + `nomic-embed-text` | ~50 ms |

The reply is **streamed sentence by sentence**: Jarvis starts speaking the first sentence while the LLM is still writing the rest, and the next sentence is synthesized while the current one plays.

## Setup

Requirements: Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), `brew install portaudio`.

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
./scripts/download_models.sh   # Kokoro voice model (~340 MB)
uv sync
```

The Whisper model (~1.6 GB) downloads automatically on first run.

## Run

```bash
uv run jarvis                    # talk to it
uv run jarvis --text             # type instead, Jarvis still speaks
uv run jarvis --text --mute      # pure text
uv run jarvis --model jarvis     # use a different Ollama model
```

Use headphones in voice mode. The mic is ignored while Jarvis is speaking, but laptop speakers can still bleed into the next turn.

### Voice commands
- **"Remember that …"**: stores a fact directly
- **"What do you remember?"**: lists stored facts
- **"Forget everything"**: wipes memory
- **"Goodbye"**: exits

## Memory

1. **Short-term**: the last 10 exchanges, persisted, so context survives restarts.
2. **Long-term facts**: after each exchange the LLM extracts durable facts ("The user is named Risa") in the background. They are deduplicated, embedded, and the most relevant ones are added to the prompt before each reply.

Everything is stored in `data/memory.db`.

## Configuration

Every field in `jarvis/config.py` can be overridden with `JARVIS_<FIELD>`:

```bash
JARVIS_VOICE=bf_emma JARVIS_VOICE_SPEED=1.1 uv run jarvis
JARVIS_STT_MODEL=mlx-community/whisper-small-mlx uv run jarvis   # faster, less accurate
JARVIS_END_SILENCE_MS=500 uv run jarvis                          # snappier turn-taking
```

Voices: `bm_george`, `bm_lewis`, `bm_daniel`, `bm_fable`, `bf_emma`, `bf_alice`, `bf_isabella`, `bf_lily`, plus many American voices (`am_*`, `af_*`).

## Roadmap

- [x] v0: streamed voice loop + persistent memory
- [ ] v1: wake word ("Hey Jarvis"), barge-in (interrupt it mid-sentence)
- [ ] v2: episodic daily summaries
- [ ] v3: tools (open apps, files, calendar, shell with confirmation)
- [ ] Custom voice (fine-tuned Piper / cloned voice from your own recordings)
