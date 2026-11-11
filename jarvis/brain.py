"""The LLM: builds the prompt (persona + memories + history) and streams the reply
back sentence by sentence so speech can start before generation finishes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime

import ollama

from . import tools
from .config import Config
from .memory import Memory

SYSTEM_PROMPT = """You are Jarvis, a calm, capable personal AI assistant running entirely offline on {user}'s Mac.
Personality: polished, dryly witty, loyal, efficient. Light British butler manner without overdoing it.

Your replies are SPOKEN ALOUD, so:
- Keep answers short: one to three sentences unless asked for detail.
- No markdown, lists, code blocks, emojis or URLs. Plain conversational sentences.
- Write numbers and symbols the way they should be said.
- If you don't know something or can't do it offline, say so plainly.

You CAN control this Mac through your tools: open, quit and switch apps, type text, press
keyboard shortcuts, set volume, control music, open files, folders and URLs, and run AppleScript
or shell commands. When the user asks for an action, call the tool. Never claim you did something
unless a tool result confirms it. Prefer the specific tools; use run_applescript or run_shell only
when nothing else fits. After acting, confirm in a few words, e.g. "Safari is open."
Speech recognition can mishear app names; if a requested app doesn't exist, pick the closest
installed app that makes sense or ask.

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
                    think=self.cfg.llm_think, keep_alive=-1, options={"num_predict": 1, "num_ctx": self.cfg.llm_ctx})
        self.memory._embed(["warmup"], "search_query")  # load the embedding model too

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

    def _stream(self, messages: list[dict], tool_calls: list) -> Iterator[str]:
        """Stream reply text; any tool calls the model makes are appended to `tool_calls`."""
        stream = ollama.chat(
            model=self.cfg.llm_model,
            messages=messages,
            tools=tools.TOOLS if self.cfg.tools_enabled else None,
            stream=True,
            think=self.cfg.llm_think,
            keep_alive=-1,
            options={"temperature": self.cfg.llm_temperature, "num_ctx": self.cfg.llm_ctx},
        )
        in_think = False
        for chunk in stream:
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)
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

    def reply(self, user_text: str, on_tool=None) -> Iterator[str]:
        """Yield the reply one sentence at a time, running any tools the model calls in between."""
        messages = self._messages(user_text)
        for _ in range(self.cfg.max_tool_rounds):
            tool_calls: list = []
            said: list[str] = []
            for sentence in split_sentences(self._stream(messages, tool_calls)):
                said.append(sentence)
                yield sentence
            if not tool_calls:
                return
            messages.append({"role": "assistant", "content": " ".join(said), "tool_calls": tool_calls})
            for tc in tool_calls:
                name, args = tc.function.name, dict(tc.function.arguments or {})
                result = tools.call(name, args)
                if on_tool:
                    on_tool(name, args, result)
                messages.append({"role": "tool", "content": result, "tool_name": name})
        yield "I've stopped there; that was taking too many steps."
