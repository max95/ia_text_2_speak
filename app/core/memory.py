from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

class SQLiteMemory:
    def __init__(
        self,
        db_path: str = "app/data/memory.sqlite",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.client = OpenAI()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _embed(self, text: str) -> List[float]:
        if not text:
            return []
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return list(response.data[0].embedding)

    def append(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO memories (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            memory_id = cursor.lastrowid
            embedding = self._embed(content)
            if embedding:
                conn.execute(
                    """
                    INSERT INTO memory_vectors (memory_id, session_id, role, content, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, session_id, role, content, json.dumps(embedding)),
                )

    def fetch_recent(self, session_id: str, limit: int) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM memories
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def search(self, session_id: str, query: str, limit: int = 6, candidate_limit: int = 200) -> List[str]:
        if not query or limit <= 0:
            return []
        query_embedding = self._embed(query)
        if not query_embedding:
            return []
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT content, embedding
                FROM memory_vectors
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, candidate_limit),
            )
            rows = cursor.fetchall()
        scored: List[tuple[float, str]] = []
        query_norm = _vector_norm(query_embedding)
        for content, embedding_json in rows:
            embedding = json.loads(embedding_json)
            score = _cosine_similarity(query_embedding, embedding, query_norm)
            scored.append((score, content))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [content for _, content in scored[:limit]]


def _vector_norm(vector: List[float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def _cosine_similarity(query: List[float], candidate: List[float], query_norm: float) -> float:
    if not query or not candidate:
        return 0.0
    candidate_norm = _vector_norm(candidate)
    if candidate_norm == 0.0 or query_norm == 0.0:
        return 0.0
    dot = sum(q * c for q, c in zip(query, candidate))
    return dot / (query_norm * candidate_norm)
