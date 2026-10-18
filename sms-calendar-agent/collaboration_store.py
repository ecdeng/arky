from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CollaborationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS coordination_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_conversation_sid TEXT NOT NULL,
                    requester TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    date_range TEXT NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS coordination_replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER NOT NULL REFERENCES coordination_requests(id),
                    phone TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    response TEXT,
                    responded_at TEXT,
                    UNIQUE(request_id, phone)
                );
                CREATE INDEX IF NOT EXISTS coordination_replies_phone_pending
                    ON coordination_replies(phone, responded_at);
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_sid TEXT PRIMARY KEY,
                    claimed_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def claim_message(self, message_sid: str) -> bool:
        """Atomically claim a Twilio message so dual webhooks/retries don't repeat it."""
        if not message_sid:
            return True
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO processed_messages(message_sid, claimed_at) VALUES (?, ?)",
                (message_sid, _utc_now()),
            )
            return cursor.rowcount == 1

    def start_request(
        self,
        parent_conversation_sid: str,
        requester: str,
        purpose: str,
        date_range: str,
        duration_minutes: int,
        participants: list[dict[str, str]],
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO coordination_requests
                    (parent_conversation_sid, requester, purpose, date_range,
                     duration_minutes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    parent_conversation_sid,
                    requester,
                    purpose,
                    date_range,
                    duration_minutes,
                    _utc_now(),
                ),
            )
            request_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO coordination_replies
                    (request_id, phone, display_name)
                VALUES (?, ?, ?)
                """,
                [
                    (request_id, person["phone"], person["name"])
                    for person in participants
                ],
            )

        return {
            "request_id": request_id,
            "parent_conversation_sid": parent_conversation_sid,
            "requester": requester,
            "purpose": purpose,
            "date_range": date_range,
            "duration_minutes": duration_minutes,
            "participants": participants,
        }

    def record_reply(self, phone: str, response: str) -> dict[str, Any] | None:
        """Record a participant response against their newest unanswered request."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT r.id AS request_id, r.parent_conversation_sid, r.requester,
                       r.purpose, r.date_range, r.duration_minutes, p.phone,
                       p.display_name
                FROM coordination_replies p
                JOIN coordination_requests r ON r.id = p.request_id
                WHERE p.phone = ? AND p.responded_at IS NULL AND r.completed_at IS NULL
                ORDER BY r.id DESC
                LIMIT 1
                """,
                (phone,),
            ).fetchone()
            if not row:
                return None

            connection.execute(
                "UPDATE coordination_replies SET response = ?, responded_at = ? "
                "WHERE request_id = ? AND phone = ?",
                (response, _utc_now(), row["request_id"], phone),
            )
            participants = connection.execute(
                """
                SELECT phone, display_name, response, responded_at
                FROM coordination_replies WHERE request_id = ? ORDER BY id
                """,
                (row["request_id"],),
            ).fetchall()
            complete = all(item["responded_at"] is not None for item in participants)
            if complete:
                connection.execute(
                    "UPDATE coordination_requests SET completed_at = ? WHERE id = ?",
                    (_utc_now(), row["request_id"]),
                )

        return {
            "request_id": row["request_id"],
            "parent_conversation_sid": row["parent_conversation_sid"],
            "requester": row["requester"],
            "purpose": row["purpose"],
            "date_range": row["date_range"],
            "duration_minutes": row["duration_minutes"],
            "complete": complete,
            "participants": [dict(item) for item in participants],
        }

    def mark_delivery_failed(self, request_id: int, phone: str) -> None:
        """Keep an undeliverable participant from blocking the whole coordination request."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE coordination_replies SET response = ?, responded_at = ? "
                "WHERE request_id = ? AND phone = ? AND responded_at IS NULL",
                ("Could not deliver the availability request.", _utc_now(), request_id, phone),
            )
            pending = connection.execute(
                "SELECT COUNT(*) FROM coordination_replies "
                "WHERE request_id = ? AND responded_at IS NULL",
                (request_id,),
            ).fetchone()[0]
            if pending == 0:
                connection.execute(
                    "UPDATE coordination_requests SET completed_at = ? WHERE id = ?",
                    (_utc_now(), request_id),
                )

    def pending_request_for(self, phone: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT r.id AS request_id, r.parent_conversation_sid, r.purpose,
                       r.date_range, r.duration_minutes, p.display_name
                FROM coordination_replies p
                JOIN coordination_requests r ON r.id = p.request_id
                WHERE p.phone = ? AND p.responded_at IS NULL AND r.completed_at IS NULL
                ORDER BY r.id DESC LIMIT 1
                """,
                (phone,),
            ).fetchone()
        return dict(row) if row else None

    def get_request(self, request_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM coordination_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if not row:
                return None
            participants = connection.execute(
                """
                SELECT phone, display_name, response, responded_at
                FROM coordination_replies WHERE request_id = ? ORDER BY id
                """,
                (request_id,),
            ).fetchall()
        result = dict(row)
        result["participants"] = [dict(item) for item in participants]
        return result
