"""Persistent memory.

* turns: every exchange, so short-term context survives restarts
* facts: durable things learned about the user, stored with embeddings and
  recalled by semantic similarity before each reply
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import ollama

EXTRACT_PROMPT = """You maintain long-term memory for a personal assistant.
From the exchange below, extract durable facts worth remembering about the user:
name, preferences, projects, people, plans, habits, important dates.
Ignore small talk, questions, and anything only relevant right now.
Write each fact as a short standalone sentence in third person ("The user ...").
Respond with JSON only: {"facts": ["...", "..."]}  (empty list if nothing new)

Already known:
%s

Exchange:
User: %s
Assistant: %s"""


class Memory:
    def __init__(self, db_path: Path, embed_model: str, llm_model: str):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.embed_model = embed_model
        self.llm_model = llm_model
        with self.lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY, text TEXT NOT NULL, embedding BLOB NOT NULL, ts REAL NOT NULL);
                """
            )
        self._load_facts()

    # ---------- embeddings ----------
    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        # nomic-embed-text expects task prefixes
        prefixed = [f"{kind}: {t}" for t in texts]
        vecs = np.array(ollama.embed(model=self.embed_model, input=prefixed, keep_alive=-1).embeddings, dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def _load_facts(self) -> None:
        with self.lock:
            rows = self.db.execute("SELECT id, text, embedding FROM facts ORDER BY id").fetchall()
        self.fact_ids = [r[0] for r in rows]
        self.fact_texts = [r[1] for r in rows]
        self.fact_vecs = (
            np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 768), np.float32)
        )

    # ---------- turns ----------
    def add_turn(self, role: str, content: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO turns (role, content, ts) VALUES (?, ?, ?)", (role, content, time.time()))
            self.db.commit()

    def recent_turns(self, n_pairs: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (n_pairs * 2,)
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # ---------- facts ----------
    def recall(self, query: str, k: int, min_score: float) -> list[str]:
        if not self.fact_texts or not query.strip():
            return []
        q = self._embed([query], "search_query")[0]
        scores = self.fact_vecs @ q
        best = np.argsort(-scores)[:k]
        return [self.fact_texts[i] for i in best if scores[i] >= min_score]

    def remember(self, fact: str, dedupe_threshold: float = 0.9) -> bool:
        fact = fact.strip()
        if not fact:
            return False
        vec = self._embed([fact], "search_document")[0]
        if len(self.fact_vecs) and float(np.max(self.fact_vecs @ vec)) >= dedupe_threshold:
            return False
        with self.lock:
            self.db.execute(
                "INSERT INTO facts (text, embedding, ts) VALUES (?, ?, ?)", (fact, vec.tobytes(), time.time())
            )
            self.db.commit()
        self._load_facts()
        return True

    def forget_all(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM facts")
            self.db.execute("DELETE FROM turns")
            self.db.commit()
        self._load_facts()

    def all_facts(self) -> list[str]:
        return list(self.fact_texts)

    def learn_from(self, user: str, assistant: str) -> list[str]:
        """Ask the LLM for durable facts in this exchange and store the new ones."""
        known = "\n".join(f"- {f}" for f in self.recall(user, 10, 0.4)) or "(nothing yet)"
        resp = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": EXTRACT_PROMPT % (known, user, assistant)}],
            format="json",
            think=False,
            keep_alive=-1,
            options={"temperature": 0},
        )
        try:
            facts = json.loads(resp.message.content).get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            return []
        return [f for f in facts if isinstance(f, str) and self.remember(f)]

    def learn_in_background(self, user: str, assistant: str, on_learned=None) -> None:
        def run():
            try:
                learned = self.learn_from(user, assistant)
                if learned and on_learned:
                    on_learned(learned)
            except Exception as e:
                print(f"\n[memory error] {e}")

        threading.Thread(target=run, daemon=True).start()
