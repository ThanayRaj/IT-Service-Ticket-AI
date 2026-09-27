import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.database import TicketDatabase


class TicketDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "nested" / "test.sqlite3"
        self.database = TicketDatabase(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initializes_database_independent_of_working_directory(self):
        self.assertTrue(self.database_path.is_file())
        self.assertTrue(self.database_path.parent.is_dir())

    def test_schema_has_required_columns_and_constraints(self):
        connection = sqlite3.connect(str(self.database_path))
        try:
            columns = {row[1]: row for row in connection.execute("PRAGMA table_info(tickets)")}
        finally:
            connection.close()
        self.assertEqual(set(columns), {
            "id", "body", "predicted_department", "predicted_priority", "created_at",
            "source", "source_key", "batch_id", "batch_row", "metadata_json",
        })
        self.assertEqual(columns["id"][5], 1)  # primary key
        for name in ("body", "predicted_department", "predicted_priority", "created_at", "source"):
            self.assertEqual(columns[name][3], 1)  # NOT NULL

    def test_insert_and_retrieve_by_id(self):
        inserted = self.database.insert_ticket(
            "VPN is unavailable", "IT Support", "high"
        )
        fetched = self.database.get_ticket_by_id(inserted["id"])
        self.assertEqual(inserted, fetched)
        self.assertEqual(inserted["body"], "VPN is unavailable")
        self.assertEqual(inserted["predicted_department"], "IT Support")
        self.assertEqual(inserted["predicted_priority"], "high")
        self.assertTrue(inserted["created_at"].endswith("Z"))
        datetime.strptime(inserted["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")

    def test_retrieves_multiple_records_and_recent_subset(self):
        first = self.database.insert_ticket("Body one", "IT Support", "high")
        second = self.database.insert_ticket("Body two", "Billing", "medium")
        third = self.database.insert_ticket("Body three", "Customer Service", "low")
        all_tickets = self.database.get_tickets()
        recent = self.database.get_recent_tickets(limit=2)
        self.assertEqual(len(all_tickets), 3)
        self.assertEqual([item["id"] for item in all_tickets],
                         [third["id"], second["id"], first["id"]])
        self.assertEqual([item["id"] for item in recent],
                         [third["id"], second["id"]])

    def test_aggregate_counts(self):
        self.database.insert_ticket("One", "IT Support", "high")
        self.database.insert_ticket("Two", "IT Support", "medium")
        self.database.insert_ticket("Three", "Billing", "high")
        self.assertEqual(self.database.get_aggregate_counts(), {
            "total_tickets": 3,
            "department_counts": {"Billing": 1, "IT Support": 2},
            "priority_counts": {"high": 2, "medium": 1},
            "source_counts": {"incoming": 3},
        })

    def test_additive_migration_preserves_legacy_rows_and_marks_them_incoming(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(str(legacy_path))
        connection.execute(
            """CREATE TABLE tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                body TEXT NOT NULL,
                predicted_department TEXT NOT NULL,
                predicted_priority TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO tickets (body, predicted_department, predicted_priority, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("Existing ticket", "IT Support", "medium", "2026-01-01T00:00:00.000Z"),
        )
        connection.commit()
        connection.close()

        migrated = TicketDatabase(legacy_path)
        record = migrated.get_ticket_by_id(1)
        self.assertEqual(record["body"], "Existing ticket")
        self.assertEqual(record["source"], "incoming")
        self.assertIsNone(record["source_key"])

    def test_bulk_batch_rows_and_metadata_persist_and_group_in_upload_order(self):
        first = self.database.insert_ticket(
            "Second row body", "IT Support", "low", source="bulk",
            source_key="bulk:abc:1", batch_id="abc", batch_row=1,
            metadata={"Department": "metadata", "Customer": "A"},
        )
        self.database.insert_ticket(
            "First row body", "Billing", "medium", source="bulk",
            source_key="bulk:abc:0", batch_id="abc", batch_row=0,
            metadata={"Customer": "B"},
        )
        reopened = TicketDatabase(self.database_path)
        rows = reopened.get_bulk_batch("abc")
        self.assertEqual([row["body"] for row in rows], ["First row body", "Second row body"])
        self.assertEqual(rows[1]["metadata"], {"Department": "metadata", "Customer": "A"})
        self.assertEqual(reopened.get_latest_bulk_batch(), "abc")
        self.assertEqual(reopened.get_ticket_by_id(first["id"])["batch_id"], "abc")

    def test_demo_records_are_counted_identified_and_only_demo_can_be_cleared(self):
        normal = self.database.insert_ticket("Normal", "IT Support", "medium", source="api")
        demo = self.database.insert_ticket(
            "Sample", "Billing", "low", source="demo", source_key="demo-ticket-01"
        )
        self.assertEqual(self.database.count_tickets_by_source("demo"), 1)
        self.assertEqual(self.database.get_ticket_by_source_key("demo-ticket-01"), demo)
        self.assertEqual(self.database.get_aggregate_counts()["source_counts"],
                         {"api": 1, "demo": 1})
        with self.assertRaises(ValueError):
            self.database.clear_tickets_by_source("api")
        self.assertEqual(self.database.clear_tickets_by_source("demo"), 1)
        self.assertIsNone(self.database.get_ticket_by_id(demo["id"]))
        self.assertEqual(self.database.get_ticket_by_id(normal["id"])["source"], "api")

    def test_duplicate_source_keys_are_rejected_by_sqlite(self):
        self.database.insert_ticket("One", "IT Support", "low", source="demo", source_key="k1")
        with self.assertRaises(sqlite3.IntegrityError):
            self.database.insert_ticket("Two", "Billing", "high", source="demo", source_key="k1")

    def test_data_persists_across_database_instances(self):
        expected = self.database.insert_ticket("Persistent ticket", "IT Support", "low")
        reopened = TicketDatabase(self.database_path)
        self.assertEqual(reopened.get_ticket_by_id(expected["id"]), expected)

    def test_invalid_empty_and_whitespace_values_are_rejected(self):
        for values in (("", "IT Support", "high"),
                       ("  ", "IT Support", "high"),
                       ("Body", "", "high"),
                       ("Body", "IT Support", " \t")):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.database.insert_ticket(*values)
        with self.assertRaises(TypeError):
            self.database.insert_ticket(None, "IT Support", "high")
        self.assertEqual(self.database.get_tickets(), [])

    def test_retrieval_validation_and_missing_id(self):
        self.assertIsNone(self.database.get_ticket_by_id(1))
        with self.assertRaises(ValueError):
            self.database.get_recent_tickets(limit=0)
        with self.assertRaises(ValueError):
            self.database.get_ticket_by_id(-1)


if __name__ == "__main__":
    unittest.main()
