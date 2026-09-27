"""Read and validate the supplied ticket dataset without changing its source."""

import csv
from pathlib import Path
from typing import Dict, List, Optional


REQUIRED_COLUMNS = ("Body", "Department", "Priority", "Tags")


class DatasetError(ValueError):
    """Raised when the source CSV is missing or does not match its expected schema."""


def load_tickets(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load CSV rows and validate required business columns.

    The supplied empty-header index column is preserved in each returned row so
    inspection remains faithful to the source, but callers should never model it.
    """
    source = Path(path) if path is not None else (
        Path(__file__).resolve().parents[1] / "data" / "IT Support Ticket Data.csv"
    )
    if not source.is_file():
        raise DatasetError("Ticket dataset was not found: {0}".format(source))
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames or []
            missing = [column for column in REQUIRED_COLUMNS if column not in headers]
            if missing:
                raise DatasetError(
                    "Ticket dataset is missing required columns: {0}".format(
                        ", ".join(missing)
                    )
                )
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError("Could not read ticket dataset: {0}".format(exc))
