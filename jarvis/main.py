"""Entry point: listen -> transcribe -> think -> speak, with memory.

    uv run jarvis            # voice mode
    uv run jarvis --text     # type instead of talk (Jarvis still speaks)
    uv run jarvis --text --mute
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from . import tools
from .config import Config
from .memory import Memory

YES_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "affirmative", "go", "do", "proceed", "please", "run"}
NO_WORDS = {"no", "nope", "don't", "dont", "stop", "cancel", "negative", "wait"}
EXIT_PHRASES = {"goodbye", "goodbye jarvis", "shut down", "exit", "quit", "go to sleep"}
DIM, CYAN, GREEN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"


def status(msg: str) -> None:
    print(f"{DIM}{msg}{RESET}", flush=True)


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def handle_command(text: str, memory: Memory) -> str | None:
    """Built-in memory commands that bypass the LLM. Returns a spoken reply or None."""
    t = normalize(text)
    m = re.match(r"^(?:jarvis\s+)?remember (?:that )?(.+)", text.strip(), flags=re.I)
    if m:
        fact = m.group(1).strip().rstrip(".")
        return "Noted." if memory.remember(f"The user said: {fact}.") else "I already knew that."
    if t in {"what do you remember", "what do you know about me", "list memories"}:
        facts = memory.all_facts()
        if not facts:
            return "I don't have anything stored about you yet."
        print("\n".join(f"  • {f}" for f in facts))
        return f"I have {len(facts)} things stored. They're on screen."
    if t in {"forget everything", "wipe your memory"}:
        memory.forget_all()
        return "Done. My memory is a clean slate."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline real-time voice assistant")
    parser.add_argument("--text", action="store_true", help="type instead of using the microphone")
    parser.add_argument("--mute", action="store_true", help="print replies without speaking them")
    parser.add_argument("--model", help="Ollama model to use (overrides JARVIS_LLM_MODEL)")
    args = parser.parse_args()

    cfg = Config()
    if args.model:
        cfg.llm_model = args.model

    status(f"Loading memory ({cfg.db_path.name}) ...")
    memory = Memory(cfg.db_path, cfg.embed_model, cfg.llm_model, cfg.llm_ctx)

    from .brain import Brain

    brain = Brain(cfg, memory)
    status(f"Warming up brain ({cfg.llm_model}) ...")
    brain.warmup()

    speaker = None
    if not args.mute:
        from .tts import Speaker

        status(f"Loading voice ({cfg.voice}) ...")
        speaker = Speaker(cfg.tts_model, cfg.tts_voices, cfg.voice, cfg.voice_lang, cfg.voice_speed)
        speaker.warmup()

    listener = transcriber = None
    if not args.text:
        from .audio import Listener
        from .stt import Transcriber

        status(f"Loading ears ({cfg.stt_model.split('/')[-1]}) ...")
        transcriber = Transcriber(cfg.stt_model, cfg.language)
        transcriber.warmup()
        listener = Listener(cfg.vad_aggressiveness, cfg.end_silence_ms, cfg.max_utterance_s)
        listener.__enter__()

    def say(text: str) -> None:
        print(f"{CYAN}Jarvis:{RESET} {text}")
        if speaker:
            speaker.say(text)
            speaker.wait()

    def on_learned(facts: list[str]) -> None:
        for f in facts:
            print(f"{DIM}  [memory +] {f}{RESET}", flush=True)

    def confirm(question: str) -> bool:
        """Ask a yes/no question out loud (or in the terminal) before a risky action."""
        title, _, detail = question.partition("\n")
        print(f"\n{YELLOW}⚠ {title}{RESET}")
        if detail:
            print(f"{DIM}{detail}{RESET}")
        if not listener:
            return input(f"{YELLOW}  [y/N]{RESET} ").strip().lower() in {"y", "yes"}
        if speaker:
            speaker.wait()
            speaker.say(title + " Yes or no?")
            speaker.wait()
        listener.flush()
        print(f"{GREEN}● waiting for yes/no...{RESET}", end="\r", flush=True)
        answer = transcriber.transcribe(listener.listen())
        words = set(re.findall(r"[a-z']+", answer.lower()))
        approved = bool(words & YES_WORDS) and not (words & NO_WORDS)
        print(f"{GREEN}You:{RESET} {answer}  → {'approved' if approved else 'declined'}")
        return approved

    def on_tool(name: str, args: dict, result: str) -> None:
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"\n{DIM}  [tool] {name}({shown}) → {result[:200]}{RESET}", flush=True)

    tools.set_confirm(confirm, cfg.confirm_risky)

    say(f"Online and at your service, {cfg.user_name}.")
    try:
        while True:
            # ---- hear ----
            if listener:
                listener.flush()  # discard anything picked up while Jarvis was talking
                print(f"{GREEN}● listening...{RESET}", end="\r", flush=True)
                audio = listener.listen()
                t0 = time.perf_counter()
                print(f"{YELLOW}◐ transcribing...{RESET}", end="\r", flush=True)
                user_text = transcriber.transcribe(audio)
                if not user_text:
                    continue
                print(f"{GREEN}You:{RESET} {user_text}" + " " * 10)
            else:
                try:
                    user_text = input(f"{GREEN}You:{RESET} ").strip()
                except EOFError:
                    break
                if not user_text:
                    continue
                t0 = time.perf_counter()

            if normalize(user_text) in EXIT_PHRASES:
                say("Very good. Powering down.")
                break

            builtin = handle_command(user_text, memory)
            if builtin:
                say(builtin)
                continue

            # ---- think + speak (streamed) ----
            print(f"{CYAN}Jarvis:{RESET} ", end="", flush=True)
            reply_parts: list[str] = []
            first = True
            for sentence in brain.reply(user_text, on_tool):
                if first:
                    latency = time.perf_counter() - t0
                    first = False
                print(sentence, end=" ", flush=True)
                reply_parts.append(sentence)
                if speaker:
                    speaker.say(sentence)
            print(f"{DIM}  ({latency:.2f}s to first sentence){RESET}" if not first else "")
            if speaker:
                speaker.wait()

            reply = " ".join(reply_parts)
            memory.add_turn("user", user_text)
            memory.add_turn("assistant", reply)
            # Short replies like "yes" / "thanks" carry no facts; skip so the LLM stays free.
            if cfg.learn_facts and len(user_text.split()) >= 4:
                memory.learn_in_background(user_text, reply, on_learned)
    except KeyboardInterrupt:
        print()
        status("Interrupted.")
    finally:
        if listener:
            listener.__exit__(None, None, None)
    sys.exit(0)


if __name__ == "__main__":
    main()
