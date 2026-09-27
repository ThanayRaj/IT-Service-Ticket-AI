"""SQLite storage for tickets and their generated classification labels."""

import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "application_tickets.sqlite3"


class TicketDatabase:
    """Small SQLite repository for predictions created by the application."""

    def __init__(self, database_path: Optional[Union[str, Path]] = None) -> None:
        self.database_path = Path(database_path or DEFAULT_DATABASE_PATH).expanduser().resolve()
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the database folder and schema if they are not present."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    body TEXT NOT NULL CHECK (length(trim(body)) > 0),
                    predicted_department TEXT NOT NULL
                        CHECK (length(trim(predicted_department)) > 0),
                    predicted_priority TEXT NOT NULL
                        CHECK (length(trim(predicted_priority)) > 0),
                    created_at TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    source TEXT NOT NULL DEFAULT 'incoming'
                        CHECK (source IN ('incoming', 'api', 'bulk', 'demo')),
                    source_key TEXT,
                    batch_id TEXT,
                    batch_row INTEGER,
                    metadata_json TEXT
                )"""
            )
            # Additive migration keeps all previously saved application tickets.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
            if "source" not in columns:
                connection.execute(
                    "ALTER TABLE tickets ADD COLUMN source TEXT NOT NULL DEFAULT 'incoming'"
                )
            if "source_key" not in columns:
                connection.execute("ALTER TABLE tickets ADD COLUMN source_key TEXT")
            for name, declaration in (("batch_id", "TEXT"), ("batch_row", "INTEGER"),
                                      ("metadata_json", "TEXT")):
                if name not in columns:
                    connection.execute("ALTER TABLE tickets ADD COLUMN {0} {1}".format(
                        name, declaration
                    ))
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_source_key "
                "ON tickets (source_key) WHERE source_key IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tickets_created_at "
                "ON tickets (created_at DESC, id DESC)"
            )

    @staticmethod
    def _validate_text(value: str, field_name: str) -> None:
        if not isinstance(value, str):
            raise TypeError("{0} must be a string".format(field_name))
        if not value.strip():
            raise ValueError("{0} cannot be empty or whitespace-only".format(field_name))

    @staticmethod
    def _as_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
        return dict(row) if row is not None else None

    def insert_ticket(
        self, body: str, predicted_department: str, predicted_priority: str,
        source: str = "incoming", source_key: Optional[str] = None,
        batch_id: Optional[str] = None, batch_row: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Insert a newly predicted ticket and return its persisted record."""
        self._validate_text(body, "body")
        self._validate_text(predicted_department, "predicted_department")
        self._validate_text(predicted_priority, "predicted_priority")
        if source not in {"incoming", "api", "bulk", "demo"}:
            raise ValueError("source must be incoming, api, bulk, or demo")
        if source_key is not None:
            self._validate_text(source_key, "source_key")
        if batch_id is not None:
            self._validate_text(batch_id, "batch_id")
        if batch_row is not None and (not isinstance(batch_row, int) or batch_row < 0):
            raise ValueError("batch_row must be a non-negative integer")
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO tickets
                   (body, predicted_department, predicted_priority, source, source_key,
                    batch_id, batch_row, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (body, predicted_department, predicted_priority, source, source_key,
                 batch_id, batch_row, metadata_json),
            )
            row = connection.execute(
                "SELECT id, body, predicted_department, predicted_priority, created_at, "
                "source, source_key, batch_id, batch_row, metadata_json "
                "FROM tickets WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return self._as_dict(row)

    def get_tickets(self, limit: Optional[int] = None, offset: int = 0,
                    source: Optional[str] = None) -> List[dict]:
        """Return tickets ordered newest first, optionally paginated."""
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        query = ("SELECT id, body, predicted_department, predicted_priority, created_at, "
                 "source, source_key, batch_id, batch_row, metadata_json FROM tickets")
        parameters = []
        if source is not None:
            query += " WHERE source = ?"
            parameters.append(source)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            parameters.extend((limit, offset))
        elif offset:
            query += " LIMIT -1 OFFSET ?"
            parameters.append(offset)
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
            return [dict(row) for row in rows]

    def get_ticket_by_id(self, ticket_id: int) -> Optional[dict]:
        """Return a ticket by primary key, or None if no such ticket exists."""
        if not isinstance(ticket_id, int) or isinstance(ticket_id, bool) or ticket_id < 1:
            raise ValueError("ticket_id must be a positive integer")
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, body, predicted_department, predicted_priority, created_at,
                          source, source_key, batch_id, batch_row, metadata_json
                   FROM tickets WHERE id = ?""",
                (ticket_id,),
            ).fetchone()
            return self._as_dict(row)

    def get_recent_tickets(self, limit: int = 20, source: Optional[str] = None) -> List[dict]:
        """Return up to ``limit`` tickets, newest first."""
        return self.get_tickets(limit=limit, source=source)

    def count_tickets_by_source(self, source: str) -> int:
        """Return the number of tickets carrying a given origin marker."""
        with self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM tickets WHERE source = ?", (source,)
            ).fetchone()[0]

    def get_ticket_by_source_key(self, source_key: str) -> Optional[dict]:
        """Return the ticket associated with an idempotency/source key."""
        with self._connection() as connection:
            row = connection.execute(
                """SELECT id, body, predicted_department, predicted_priority, created_at,
                          source, source_key, batch_id, batch_row, metadata_json
                   FROM tickets WHERE source_key = ?""",
                (source_key,),
            ).fetchone()
            return self._as_dict(row)

    def get_bulk_batch(self, batch_id: str) -> List[dict]:
        """Read stored rows from one processed CSV upload in original row order."""
        self._validate_text(batch_id, "batch_id")
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, body, predicted_department, predicted_priority, created_at,
                          source, source_key, batch_id, batch_row, metadata_json
                   FROM tickets WHERE source = 'bulk' AND batch_id = ?
                   ORDER BY batch_row, id""", (batch_id,)
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["metadata"] = json.loads(record.pop("metadata_json") or "{}")
            records.append(record)
        return records

    def get_latest_bulk_batch(self) -> Optional[str]:
        """Return the most recently written persisted bulk upload identifier."""
        with self._connection() as connection:
            row = connection.execute(
                """SELECT batch_id FROM tickets
                   WHERE source = 'bulk' AND batch_id IS NOT NULL
                   GROUP BY batch_id ORDER BY MAX(created_at) DESC, MAX(id) DESC LIMIT 1"""
            ).fetchone()
            return row[0] if row else None

    def clear_tickets_by_source(self, source: str) -> int:
        """Delete only records bearing the specified source marker."""
        if source != "demo":
            raise ValueError("only demo records may be bulk-cleared")
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM tickets WHERE source = ?", (source,))
            return cursor.rowcount

    def get_aggregate_counts(self) -> Dict[str, object]:
        """Return total, Department, and Priority counts from stored records."""
        with self._connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            departments = connection.execute(
                """SELECT predicted_department, COUNT(*) AS count
                   FROM tickets GROUP BY predicted_department
                   ORDER BY predicted_department"""
            ).fetchall()
            priorities = connection.execute(
                """SELECT predicted_priority, COUNT(*) AS count
                   FROM tickets GROUP BY predicted_priority
                   ORDER BY predicted_priority"""
            ).fetchall()
            sources = connection.execute(
                "SELECT source, COUNT(*) AS count FROM tickets GROUP BY source"
            ).fetchall()
        return {
            "total_tickets": total,
            "department_counts": {row["predicted_department"]: row["count"]
                                  for row in departments},
            "priority_counts": {row["predicted_priority"]: row["count"]
                                for row in priorities},
            "source_counts": {row["source"]: row["count"] for row in sources},
        }
