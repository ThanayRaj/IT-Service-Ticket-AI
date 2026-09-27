import unittest
from datetime import datetime, timedelta, timezone

from src.sla import (
    DEFAULT_AT_RISK_FRACTION,
    DEFAULT_SLA_THRESHOLDS,
    SLAInputError,
    SLARiskEngine,
    UnsupportedPriorityError,
)


class SLARiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = SLARiskEngine()
        self.reference = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)

    def assess_age(self, priority, age):
        return self.engine.evaluate(
            self.reference - age, priority, self.reference
        )

    def test_demonstration_thresholds_are_configured_per_priority(self):
        self.assertEqual(DEFAULT_SLA_THRESHOLDS, {
            "high": timedelta(hours=4),
            "medium": timedelta(hours=24),
            "low": timedelta(hours=72),
        })
        for priority, threshold in DEFAULT_SLA_THRESHOLDS.items():
            with self.subTest(priority=priority):
                result = self.assess_age(priority, threshold / 2)
                self.assertEqual(result["status"], "Within SLA")
                self.assertEqual(result["sla_threshold_seconds"], threshold.total_seconds())

    def test_within_sla_before_risk_boundary(self):
        result = self.assess_age("high", timedelta(hours=3, minutes=11))
        self.assertEqual(result["status"], "Within SLA")

    def test_exact_at_risk_boundary_is_at_risk(self):
        boundary = DEFAULT_SLA_THRESHOLDS["high"] * DEFAULT_AT_RISK_FRACTION
        self.assertEqual(self.assess_age("high", boundary)["status"], "At Risk")

    def test_exact_sla_threshold_is_breached(self):
        for priority, threshold in DEFAULT_SLA_THRESHOLDS.items():
            with self.subTest(priority=priority):
                self.assertEqual(self.assess_age(priority, threshold)["status"], "Breached")

    def test_after_sla_threshold_is_breached(self):
        self.assertEqual(self.assess_age("low", timedelta(hours=73))["status"], "Breached")

    def test_unsupported_priority_is_rejected(self):
        with self.assertRaises(UnsupportedPriorityError):
            self.assess_age("urgent", timedelta(minutes=1))

    def test_missing_or_invalid_timestamp_is_rejected(self):
        with self.assertRaisesRegex(SLAInputError, "ticket_created_at is required"):
            self.engine.evaluate(None, "high", self.reference)
        with self.assertRaisesRegex(SLAInputError, "reference_time is required"):
            self.engine.evaluate(self.reference, "high", None)
        for timestamp in ("", "not-a-timestamp", "2026-09-26T12:00:00"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(SLAInputError):
                    self.engine.evaluate(timestamp, "high", self.reference)

    def test_future_created_time_is_rejected(self):
        with self.assertRaisesRegex(SLAInputError, "cannot be in the future"):
            self.engine.evaluate(
                self.reference + timedelta(seconds=1), "high", self.reference
            )

    def test_offset_timestamps_are_normalized_to_utc(self):
        result = self.engine.evaluate(
            "2026-09-26T11:30:00+05:30", "high", "2026-09-26T07:00:00Z"
        )
        self.assertEqual(result["ticket_created_at_utc"], "2026-09-26T06:00:00.000000Z")
        self.assertEqual(result["reference_time_utc"], "2026-09-26T07:00:00.000000Z")
        self.assertEqual(result["elapsed_seconds"], 3600.0)

    def test_same_inputs_are_deterministic(self):
        args = (self.reference - timedelta(hours=2), "high", self.reference)
        self.assertEqual(self.engine.evaluate(*args), self.engine.evaluate(*args))

    def test_missing_priority_and_invalid_configuration_are_rejected(self):
        with self.assertRaisesRegex(SLAInputError, "predicted_priority is required"):
            self.engine.evaluate(self.reference, None, self.reference)
        with self.assertRaises(SLAInputError):
            SLARiskEngine(thresholds={"high": timedelta(hours=4)})
        with self.assertRaises(SLAInputError):
            SLARiskEngine(at_risk_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
