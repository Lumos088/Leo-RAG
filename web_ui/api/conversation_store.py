"""SQLite persistence for conversations, messages, citations, and memory summaries."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TITLE = "New conversation"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ConversationNotFound(KeyError):
    pass


class ConversationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_through_message_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    retrieval_json TEXT,
                    citations_json TEXT,
                    metadata_json TEXT,
                    request_id TEXT,
                    status TEXT NOT NULL DEFAULT 'complete',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    UNIQUE(conversation_id, request_id, role)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                    ON conversations(updated_at DESC);
                """
            )

    @staticmethod
    def _conversation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "summary": row["summary"],
            "summarized_through_message_id": row["summarized_through_message_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        def decode(value: str | None, fallback):
            if not value:
                return fallback
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return fallback

        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "retrieval": decode(row["retrieval_json"], []),
            "citations": decode(row["citations_json"], []),
            "metadata": decode(row["metadata_json"], {}),
            "request_id": row["request_id"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        conversation_id = uuid.uuid4().hex
        now = utc_now()
        clean_title = (title or DEFAULT_TITLE).strip()[:120] or DEFAULT_TITLE
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, clean_title, now, now),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        return self._conversation(row)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.created_at DESC
                """
            ).fetchall()
        output = []
        for row in rows:
            item = self._conversation(row)
            item["message_count"] = int(row["message_count"])
            output.append(item)
        return output

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        clean_title = title.strip()[:120]
        if not clean_title:
            raise ValueError("Conversation title cannot be empty")
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, now, conversation_id),
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(conversation_id)
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            if cursor.rowcount == 0:
                raise ConversationNotFound(conversation_id)

    def clear_messages(self, conversation_id: str) -> dict[str, Any]:
        self.get_conversation(conversation_id)
        now = utc_now()
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            connection.execute(
                """UPDATE conversations
                   SET summary = '', summarized_through_message_id = NULL, updated_at = ?
                   WHERE id = ?""",
                (now, conversation_id),
            )
        return self.get_conversation(conversation_id)

    def list_messages(self, conversation_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        self.get_conversation(conversation_id)
        query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC"
        params: tuple[Any, ...] = (conversation_id,)
        if limit is not None:
            query = """
                SELECT * FROM (
                    SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
            """
            params = (conversation_id, int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._message(row) for row in rows]

    def get_message_by_request(
        self, conversation_id: str, request_id: str, role: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM messages
                   WHERE conversation_id = ? AND request_id = ? AND role = ?""",
                (conversation_id, request_id, role),
            ).fetchone()
        return self._message(row) if row else None

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        retrieval: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        status: str = "complete",
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        if not content.strip():
            raise ValueError("message content cannot be empty")
        self.get_conversation(conversation_id)
        now = utc_now()
        with self._connect() as connection:
            try:
                cursor = connection.execute(
                    """INSERT INTO messages(
                           conversation_id, role, content, retrieval_json,
                           citations_json, metadata_json, request_id, status, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        conversation_id,
                        role,
                        content,
                        json.dumps(retrieval or [], ensure_ascii=False),
                        json.dumps(citations or [], ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        request_id,
                        status,
                        now,
                    ),
                )
                message_id = cursor.lastrowid
                if role == "user":
                    current = connection.execute(
                        "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
                    ).fetchone()
                    if current and current["title"] == DEFAULT_TITLE:
                        generated_title = content.strip().replace("\n", " ")[:36]
                        connection.execute(
                            "UPDATE conversations SET title = ? WHERE id = ?",
                            (generated_title, conversation_id),
                        )
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
                )
            except sqlite3.IntegrityError:
                if request_id:
                    row = connection.execute(
                        """SELECT * FROM messages
                           WHERE conversation_id = ? AND request_id = ? AND role = ?""",
                        (conversation_id, request_id, role),
                    ).fetchone()
                    if row:
                        return self._message(row)
                raise
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return self._message(row)

    def update_summary(
        self, conversation_id: str, summary: str, through_message_id: int | None
    ) -> dict[str, Any]:
        self.get_conversation(conversation_id)
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """UPDATE conversations
                   SET summary = ?, summarized_through_message_id = ?, updated_at = ?
                   WHERE id = ?""",
                (summary[:4000], through_message_id, now, conversation_id),
            )
        return self.get_conversation(conversation_id)
