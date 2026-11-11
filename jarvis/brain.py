"""The LLM: builds the prompt (persona + memories + history) and streams the reply
back sentence by sentence so speech can start before generation finishes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime

import ollama

from .config import Config
from .memory import Memory

SYSTEM_PROMPT = """You are Jarvis, a calm, capable personal AI assistant running entirely offline on {user}'s Mac.
Personality: polished, dryly witty, loyal, efficient. Light British butler manner without overdoing it.

Your replies are SPOKEN ALOUD, so:
- Keep answers short: one to three sentences unless asked for detail.
- No markdown, lists, code blocks, emojis or URLs. Plain conversational sentences.
- Write numbers and symbols the way they should be said.
- If you don't know something or can't do it offline, say so plainly.

Current date and time: {now}.
{memories}"""

# End of sentence: . ! ? followed by whitespace, or a newline.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_CHUNK = 12  # don't send tiny fragments like "Mr." to TTS on their own


def split_sentences(tokens: Iterator[str]) -> Iterator[str]:
    buf = ""
    for tok in tokens:
        buf += tok
        while True:
            m = _SENTENCE_END.search(buf, _MIN_CHUNK)
            if not m:
                break
            sentence, buf = buf[: m.start()].strip(), buf[m.end():]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


class Brain:
    def __init__(self, cfg: Config, memory: Memory):
        self.cfg = cfg
        self.memory = memory

    def warmup(self) -> None:
        ollama.chat(model=self.cfg.llm_model, messages=[{"role": "user", "content": "hi"}],
                    think=self.cfg.llm_think, keep_alive=-1, options={"num_predict": 1})

    def _messages(self, user_text: str) -> list[dict]:
        facts = self.memory.recall(user_text, self.cfg.recall_k, self.cfg.recall_min_score)
        memories = ("Things you remember about the user:\n" + "\n".join(f"- {f}" for f in facts)) if facts else ""
        system = SYSTEM_PROMPT.format(
            user=self.cfg.user_name,
            now=datetime.now().strftime("%A %d %B %Y, %H:%M"),
            memories=memories,
        )
        return (
            [{"role": "system", "content": system}]
            + self.memory.recent_turns(self.cfg.history_turns)
            + [{"role": "user", "content": user_text}]
        )

    def _tokens(self, user_text: str) -> Iterator[str]:
        stream = ollama.chat(
            model=self.cfg.llm_model,
            messages=self._messages(user_text),
            stream=True,
            think=self.cfg.llm_think,
            keep_alive=-1,
            options={"temperature": self.cfg.llm_temperature},
        )
        in_think = False
        for chunk in stream:
            text = chunk.message.content or ""
            # Safety net in case a custom model emits <think> blocks inline.
            if "<think>" in text:
                in_think, text = True, text.split("<think>", 1)[0]
            if in_think:
                if "</think>" in text:
                    in_think, text = False, text.split("</think>", 1)[1]
                else:
                    continue
            if text:
                yield text

    def reply(self, user_text: str) -> Iterator[str]:
        """Yield the reply one sentence at a time."""
        yield from split_sentences(self._tokens(user_text))
