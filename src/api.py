"""FastAPI automatic intake endpoint backed by the saved model pipelines."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictStr, field_validator

from src.service_factory import create_application_service


class TicketIntakeRequest(BaseModel):
    body: StrictStr

    @field_validator("body")
    @classmethod
    def body_must_contain_text(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("body must contain non-whitespace text")
        return value


class TicketIntakeResponse(BaseModel):
    ticket_id: int
    body: str
    predicted_department: str
    predicted_priority: str
    sla_status: str
    created_at: str


@lru_cache(maxsize=1)
def _default_application_service():
    return create_application_service()


def create_api_app(application_service=None) -> FastAPI:
    """Create an API instance; injection keeps tests isolated from real data."""
    api = FastAPI(
        title="IT Support AI Ticket Intake API",
        description=(
            "API-ready automatic intake using the project's saved Body-only "
            "classification pipelines and configurable SLA demonstration rules."
        ),
        version="1.0.0",
    )
    api.state.application_service = application_service

    @api.post("/api/tickets", response_model=TicketIntakeResponse, status_code=201)
    def create_ticket(payload: TicketIntakeRequest):
        service = api.state.application_service or _default_application_service()
        try:
            result = service.submit_ticket(payload.body, source="api")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Ticket intake is temporarily unavailable."
            ) from exc
        ticket = result["ticket"]
        return TicketIntakeResponse(
            ticket_id=ticket["id"],
            body=ticket["body"],
            predicted_department=result["prediction"]["department"],
            predicted_priority=result["prediction"]["priority"],
            sla_status=result["sla"]["status"],
            created_at=ticket["created_at"],
        )

    return api


app = create_api_app()
