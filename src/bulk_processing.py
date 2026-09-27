"""CSV inference workflow built on the shared application service."""

from typing import Callable, Optional
import uuid

import pandas as pd


class BulkInputError(ValueError):
    """Raised when an uploaded ticket table is not safe to process."""


PREDICTION_COLUMNS = (
    "predicted_department", "predicted_priority", "sla_status",
    "application_created_at",
)


def validate_bulk_frame(frame: pd.DataFrame) -> None:
    """Validate required Body input and prevent silent loss of input fields."""
    if not isinstance(frame, pd.DataFrame):
        raise BulkInputError("The uploaded CSV could not be read as a table.")
    if "Body" not in frame.columns:
        raise BulkInputError("The CSV must include a column named 'Body'.")
    collisions = sorted(set(PREDICTION_COLUMNS).intersection(frame.columns))
    if collisions:
        raise BulkInputError(
            "Rename reserved output column(s) before upload: {0}.".format(
                ", ".join(collisions)
            )
        )
    for position, value in enumerate(frame["Body"].tolist(), start=2):
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            raise BulkInputError(
                "Body on CSV row {0} must be text. Blank Body rows are allowed.".format(
                    position
                )
            )


def process_bulk_frame(
    frame: pd.DataFrame,
    application_service,
    progress_callback: Optional[Callable[[float], None]] = None,
    batch_id: Optional[str] = None,
) -> pd.DataFrame:
    """Classify and store nonblank rows, preserving every original CSV field.

    Department, Priority, Tags, and all other uploaded columns are retained as
    metadata only. The existing saved Body-only pipelines are called through
    TicketApplicationService; no fit or retraining operation occurs here.
    Blank Body rows remain in the downloaded output with blank predictions and
    do not create application tickets.
    """
    validate_bulk_frame(frame)
    batch_id = batch_id or uuid.uuid4().hex
    result = frame.copy(deep=True)
    departments = [""] * len(result)
    priorities = [""] * len(result)
    statuses = [""] * len(result)
    created_at_values = [""] * len(result)
    valid_positions = [
        i for i, value in enumerate(result["Body"].tolist())
        if isinstance(value, str) and value.strip()
    ]
    total = len(valid_positions)
    if progress_callback:
        progress_callback(0.0)

    for complete, position in enumerate(valid_positions, start=1):
        submitted = application_service.submit_ticket(
            result.iloc[position]["Body"], source="bulk",
            source_key="bulk:{0}:{1}".format(batch_id, position),
            batch_id=batch_id, batch_row=position,
            metadata={key: str(result.iloc[position][key]) for key in result.columns
                      if key != "Body"},
        )
        departments[position] = submitted["prediction"]["department"]
        priorities[position] = submitted["prediction"]["priority"]
        statuses[position] = submitted["sla"]["status"]
        created_at_values[position] = submitted["ticket"]["created_at"]
        if progress_callback:
            progress_callback(complete / total)

    result["predicted_department"] = departments
    result["predicted_priority"] = priorities
    result["sla_status"] = statuses
    result["application_created_at"] = created_at_values
    return result


def bulk_frame_from_records(records) -> pd.DataFrame:
    """Rebuild the processed CSV view from persisted ticket rows."""
    rows = []
    for record in records:
        row = {"Body": record["body"]}
        row.update(record.get("metadata", {}))
        row.update({
            "predicted_department": record["predicted_department"],
            "predicted_priority": record["predicted_priority"],
            "sla_status": "",
            "application_created_at": record["created_at"],
        })
        rows.append(row)
    return pd.DataFrame(rows)
