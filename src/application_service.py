"""Coordinate prediction, ticket persistence, and SLA status for the app."""

from collections import Counter
from datetime import datetime, timezone
from typing import Optional


class TicketApplicationService:
    """Thin orchestration layer around the existing backend components."""

    def __init__(self, prediction_service, database, sla_engine, clock=None):
        self.prediction_service = prediction_service
        self.database = database
        self.sla_engine = sla_engine
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _reference_time(self, reference_time):
        return self.clock() if reference_time is None else reference_time

    def submit_ticket(self, body: str, reference_time=None, source="incoming",
                      source_key=None, batch_id=None, batch_row=None, metadata=None) -> dict:
        """Predict, persist, then assess SLA from the persisted UTC timestamp."""
        prediction = self.prediction_service.predict(body)
        ticket = self.database.insert_ticket(
            body, prediction["department"], prediction["priority"],
            source=source, source_key=source_key, batch_id=batch_id,
            batch_row=batch_row, metadata=metadata,
        )
        sla = self.sla_engine.evaluate(
            ticket["created_at"], prediction["priority"],
            self._reference_time(reference_time),
        )
        return {"ticket": ticket, "prediction": prediction, "sla": sla}

    def get_ticket(self, ticket_id: int, reference_time=None) -> Optional[dict]:
        """Retrieve one stored ticket and compute its current rule-based SLA status."""
        ticket = self.database.get_ticket_by_id(ticket_id)
        if ticket is None:
            return None
        sla = self.sla_engine.evaluate(
            ticket["created_at"], ticket["predicted_priority"],
            self._reference_time(reference_time),
        )
        return {"ticket": ticket, "sla": sla}

    def get_ticket_history(self, limit: int = 200, reference_time=None) -> list:
        """Return recent application tickets enriched with their current SLA status."""
        reference = self._reference_time(reference_time)
        tickets = self.database.get_recent_tickets(limit=limit)
        history = []
        for ticket in tickets:
            sla = self.sla_engine.evaluate(
                ticket["created_at"], ticket["predicted_priority"], reference
            )
            history.append(dict(ticket, sla_status=sla["status"]))
        return history

    def get_analytics(self, reference_time=None) -> dict:
        """Return database aggregates and SLA counts from actual stored records."""
        aggregates = self.database.get_aggregate_counts()
        reference = self._reference_time(reference_time)
        sla_counts = Counter()
        for ticket in self.database.get_tickets():
            assessment = self.sla_engine.evaluate(
                ticket["created_at"], ticket["predicted_priority"], reference
            )
            sla_counts[assessment["status"]] += 1
        aggregates["sla_status_counts"] = dict(sla_counts)
        return aggregates
