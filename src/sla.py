"""Transparent, configurable SLA status rules for application-created tickets.

These demonstration thresholds are assumptions for the application only. They
are not learned from the supplied dataset and do not represent a company policy.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Mapping, Optional, Union


Timestamp = Union[str, datetime]

# Demonstration values only; change these in configuration when appropriate.
DEFAULT_SLA_THRESHOLDS = {
    "high": timedelta(hours=4),
    "medium": timedelta(hours=24),
    "low": timedelta(hours=72),
}
DEFAULT_AT_RISK_FRACTION = 0.8
SUPPORTED_PRIORITIES = frozenset(DEFAULT_SLA_THRESHOLDS)


class SLAInputError(ValueError):
    """Raised when timestamp or SLA configuration input is invalid."""


class UnsupportedPriorityError(ValueError):
    """Raised when a priority has no configured SLA threshold."""


def _parse_utc_timestamp(value: Optional[Timestamp], field_name: str) -> datetime:
    if value is None:
        raise SLAInputError("{0} is required".format(field_name))
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            timestamp_text = value.strip()
            if not timestamp_text:
                raise ValueError("timestamp is empty")
            if timestamp_text.endswith(("Z", "z")):
                timestamp_text = timestamp_text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(timestamp_text)
        except ValueError as exc:
            raise SLAInputError("{0} must be a valid ISO 8601 timestamp: {1}".format(
                field_name, exc
            )) from exc
    else:
        raise SLAInputError("{0} must be a datetime or ISO 8601 string".format(field_name))

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SLAInputError("{0} must include a timezone".format(field_name))
    return parsed.astimezone(timezone.utc)


def _format_utc_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


class SLARiskEngine:
    """Calculate an SLA status from ticket age and explicit configured rules."""

    def __init__(
        self,
        thresholds: Optional[Mapping[str, timedelta]] = None,
        at_risk_fraction: float = DEFAULT_AT_RISK_FRACTION,
    ) -> None:
        configured = dict(thresholds if thresholds is not None else DEFAULT_SLA_THRESHOLDS)
        if set(configured) != SUPPORTED_PRIORITIES:
            raise SLAInputError(
                "thresholds must define exactly: {0}".format(
                    ", ".join(sorted(SUPPORTED_PRIORITIES))
                )
            )
        for priority, duration in configured.items():
            if not isinstance(duration, timedelta) or duration.total_seconds() <= 0:
                raise SLAInputError(
                    "threshold for {0} must be a positive timedelta".format(priority)
                )
        if (isinstance(at_risk_fraction, bool)
                or not isinstance(at_risk_fraction, (int, float))
                or not math.isfinite(at_risk_fraction)
                or not 0 <= at_risk_fraction < 1):
            raise SLAInputError("at_risk_fraction must be a finite number from 0 inclusive to 1 exclusive")

        self.thresholds = configured
        self.at_risk_fraction = float(at_risk_fraction)

    def evaluate(
        self,
        ticket_created_at: Optional[Timestamp],
        predicted_priority: Optional[str],
        reference_time: Optional[Timestamp],
    ) -> Dict[str, object]:
        """Return status and the rule inputs/calculations in normalized UTC.

        `At Risk` begins when elapsed time reaches the configured fraction of
        the priority threshold. `Breached` begins at the exact threshold.
        """
        if predicted_priority is None:
            raise SLAInputError("predicted_priority is required")
        if not isinstance(predicted_priority, str):
            raise UnsupportedPriorityError("predicted_priority must be a string")
        priority = predicted_priority.strip().casefold()
        if priority not in self.thresholds:
            raise UnsupportedPriorityError(
                "Unsupported priority {0!r}; expected one of: {1}".format(
                    predicted_priority, ", ".join(sorted(self.thresholds))
                )
            )

        created = _parse_utc_timestamp(ticket_created_at, "ticket_created_at")
        reference = _parse_utc_timestamp(reference_time, "reference_time")
        if created > reference:
            raise SLAInputError("ticket_created_at cannot be in the future of reference_time")

        threshold = self.thresholds[priority]
        elapsed = reference - created
        threshold_seconds = threshold.total_seconds()
        elapsed_seconds = elapsed.total_seconds()
        at_risk_after_seconds = threshold_seconds * self.at_risk_fraction
        if elapsed >= threshold:
            status = "Breached"
        elif elapsed_seconds >= at_risk_after_seconds:
            status = "At Risk"
        else:
            status = "Within SLA"

        return {
            "status": status,
            "predicted_priority": priority,
            "ticket_created_at_utc": _format_utc_timestamp(created),
            "reference_time_utc": _format_utc_timestamp(reference),
            "elapsed_seconds": elapsed_seconds,
            "sla_threshold_seconds": threshold_seconds,
            "at_risk_after_seconds": at_risk_after_seconds,
            "remaining_seconds": threshold_seconds - elapsed_seconds,
        }
