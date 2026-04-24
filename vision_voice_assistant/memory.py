from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class MemoryItem:
    id: int
    kind: str
    content: str
    created_at: str
    metadata: dict


class MemoryStore:
    """Persistent assistant memory backed by SQLite and FTS5.

    The store keeps observations, user facts, commands and regular memories in
    the same searchable table. Recent observations are kept separately as a
    convenience for camera workflows.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.setup()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def setup(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, kind, tokenize='unicode61', content='memories', content_rowid='id');

                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    prompt TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_memory(self, kind: str, content: str, metadata: dict | None = None) -> int:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        created_at = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO memories(kind, content, metadata_json, created_at) VALUES (?, ?, ?, ?)",
                (kind, content.strip(), metadata_json, created_at),
            )
            memory_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, kind) VALUES (?, ?, ?)",
                (memory_id, content.strip(), kind),
            )
            return memory_id

    def add_message(self, role: str, content: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages(role, content, created_at) VALUES (?, ?, ?)",
                (role, content.strip(), utc_now()),
            )
            return int(cur.lastrowid)

    def add_observation(self, image_path: str | Path | None, prompt: str, description: str) -> int:
        created_at = utc_now()
        image_value = str(image_path) if image_path else None
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO observations(image_path, prompt, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (image_value, prompt.strip(), description.strip(), created_at),
            )
            observation_id = int(cur.lastrowid)
        self.add_memory(
            "observation",
            f"Наблюдение камеры: {description.strip()}",
            {"image_path": image_value, "prompt": prompt, "observation_id": observation_id},
        )
        return observation_id

    def remember_user_text(self, text: str) -> int:
        cleaned = text.strip()
        if cleaned.lower().startswith("запомни"):
            cleaned = cleaned[7:].strip(" :,-")
        return self.add_memory("user_fact", cleaned, {"source": "voice_command"})

    def recent_memories(self, limit: int = 8) -> list[MemoryItem]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def recent_observations(self, limit: int = 5) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM observations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def retrieve(self, query: str, limit: int = 8) -> list[MemoryItem]:
        tokens = self._tokens(query)
        if not tokens:
            return self.recent_memories(limit)
        match_query = " OR ".join(tokens[:12])
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT memories.*
                    FROM memories_fts
                    JOIN memories ON memories_fts.rowid = memories.id
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25(memories_fts)
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            return self.recent_memories(limit)
        return [self._row_to_memory(row) for row in rows]

    def context_for_prompt(self, query: str, limit: int = 8) -> str:
        relevant = self.retrieve(query, limit=limit)
        recent_observations = self.recent_observations(limit=3)
        blocks: list[str] = []
        if relevant:
            blocks.append("Память ассистента:")
            for item in relevant:
                blocks.append(f"- [{item.kind} {item.created_at}] {item.content}")
        if recent_observations:
            blocks.append("Последние наблюдения камеры:")
            for row in recent_observations:
                blocks.append(f"- [{row['created_at']}] {row['description']}")
        return "\n".join(blocks).strip()

    def export_text(self, limit: int = 30) -> str:
        memories = self.recent_memories(limit=limit)
        if not memories:
            return "Память пока пустая."
        return "\n".join(f"{m.created_at} [{m.kind}] {m.content}" for m in memories)

    def _tokens(self, text: str) -> list[str]:
        tokens = [token.lower() for token in TOKEN_RE.findall(text)]
        skip = {"что", "это", "там", "на", "и", "в", "во", "как", "какой", "какая"}
        return [token for token in tokens if len(token) > 1 and token not in skip]

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return MemoryItem(
            id=int(row["id"]),
            kind=row["kind"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=metadata,
        )

    def bulk_add(self, kind: str, contents: Iterable[str]) -> int:
        count = 0
        for content in contents:
            if content.strip():
                self.add_memory(kind, content)
                count += 1
        return count

