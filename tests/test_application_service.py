import importlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.application_service import TicketApplicationService
from src.database import TicketDatabase
from src.prediction import PredictionService
from src.sla import SLARiskEngine


class ApplicationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prediction_service = PredictionService()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = TicketDatabase(Path(self.temp_dir.name) / "application-test.sqlite3")
        self.application = TicketApplicationService(
            prediction_service=self.prediction_service,
            database=database,
            sla_engine=SLARiskEngine(),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_application_module_imports_without_running_or_training(self):
        module = importlib.import_module("app.app")
        self.assertTrue(callable(module.main))

    def test_prediction_to_database_flow_and_sla_from_stored_timestamp(self):
        body = "The office VPN keeps disconnecting and blocks access to the portal."
        reference = datetime(2099, 1, 1, tzinfo=timezone.utc)
        result = self.application.submit_ticket(body, reference_time=reference)
        ticket = result["ticket"]
        self.assertEqual(ticket["body"], body)
        self.assertEqual(ticket["predicted_department"], result["prediction"]["department"])
        self.assertEqual(ticket["predicted_priority"], result["prediction"]["priority"])
        self.assertEqual(result["sla"]["status"], "Breached")

        fetched = self.application.database.get_ticket_by_id(ticket["id"])
        expected_sla = self.application.sla_engine.evaluate(
            fetched["created_at"], fetched["predicted_priority"], reference
        )
        self.assertEqual(result["sla"], expected_sla)

    def test_database_history_and_selected_ticket_details(self):
        reference = datetime(2099, 1, 1, tzinfo=timezone.utc)
        saved = self.application.submit_ticket(
            "I cannot connect to the office wireless network.", reference_time=reference
        )
        history = self.application.get_ticket_history(reference_time=reference)
        detail = self.application.get_ticket(saved["ticket"]["id"], reference_time=reference)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], saved["ticket"]["id"])
        self.assertEqual(history[0]["sla_status"], saved["sla"]["status"])
        self.assertEqual(detail["ticket"], saved["ticket"])
        self.assertEqual(detail["sla"]["status"], "Breached")

    def test_empty_database_state_has_no_fabricated_counts(self):
        self.assertEqual(self.application.get_ticket_history(), [])
        self.assertEqual(self.application.get_analytics(), {
            "total_tickets": 0,
            "department_counts": {},
            "priority_counts": {},
            "source_counts": {},
            "sla_status_counts": {},
        })


if __name__ == "__main__":
    unittest.main()
