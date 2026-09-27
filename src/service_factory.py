"""Shared construction of application-facing services."""

import os
from pathlib import Path
from typing import Optional, Union

from src.application_service import TicketApplicationService
from src.database import TicketDatabase
from src.prediction import PredictionService
from src.sla import SLARiskEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH_ENV = "IT_SUPPORT_AI_DATABASE_PATH"


def resolve_database_path(database_path: Optional[Union[str, Path]] = None):
    """Resolve an explicit path or configured path independent of cwd."""
    configured = database_path or os.environ.get(DATABASE_PATH_ENV)
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def create_application_service(database_path=None) -> TicketApplicationService:
    """Build the same prediction/database/SLA service for UI and API callers."""
    return TicketApplicationService(
        prediction_service=PredictionService(),
        database=TicketDatabase(resolve_database_path(database_path)),
        sla_engine=SLARiskEngine(),
    )
