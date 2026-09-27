import asyncio
from unittest.mock import Mock

import httpx
import pytest

from src.application_service import TicketApplicationService
from src.api import create_api_app
from src.database import TicketDatabase
from src.sla import SLARiskEngine
from src.service_factory import create_application_service


def _post(api, payload):
    async def send():
        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/tickets", json=payload)
    return asyncio.run(send())


class DeterministicPredictionStub:
    def __init__(self):
        self.seen = []

    def predict(self, body):
        self.seen.append(body)
        return {"department": "IT Support", "priority": "high"}


@pytest.fixture
def client(tmp_path):
    database = TicketDatabase(tmp_path / "api_tickets.sqlite3")
    predictor = DeterministicPredictionStub()
    service = TicketApplicationService(predictor, database, SLARiskEngine())
    return create_api_app(service), database, predictor


def test_valid_ticket_returns_predictions_sla_and_persists(client):
    api, database, predictor = client
    response = _post(api, {
        "body": "The office VPN keeps disconnecting and I cannot access the portal."
    })

    assert response.status_code == 201
    body = response.json()
    assert body["ticket_id"] == 1
    assert body["predicted_department"] == "IT Support"
    assert body["predicted_priority"] == "high"
    assert body["sla_status"] == "Within SLA"
    assert body["created_at"].endswith("Z")
    assert database.get_ticket_by_id(1)["body"] == body["body"]
    assert database.get_ticket_by_id(1)["source"] == "api"
    assert predictor.seen == [body["body"]]


@pytest.mark.parametrize("body", ["", "   ", "\n\t  "])
def test_empty_and_whitespace_bodies_are_rejected(client, body):
    api, database, predictor = client
    response = _post(api, {"body": body})
    assert response.status_code == 422
    assert database.get_tickets() == []
    assert predictor.seen == []


@pytest.mark.parametrize("payload", [{}, {"body": 42}, {"unexpected": "field"}, None])
def test_malformed_requests_are_rejected(client, payload):
    api, database, predictor = client
    response = _post(api, payload)
    assert response.status_code == 422
    assert database.get_tickets() == []
    assert predictor.seen == []


def test_same_input_has_deterministic_labels_and_distinct_persisted_records(client):
    api, database, _ = client
    payload = {"body": "The office VPN keeps disconnecting."}
    first = _post(api, payload)
    second = _post(api, payload)

    assert first.status_code == second.status_code == 201
    assert first.json()["predicted_department"] == second.json()["predicted_department"]
    assert first.json()["predicted_priority"] == second.json()["predicted_priority"]
    assert [ticket["id"] for ticket in database.get_tickets()] == [2, 1]


def test_api_intake_does_not_fit_or_retrain_models(client):
    api, _, predictor = client
    response = _post(api, {"body": "Laptop will not connect to Wi-Fi."})
    assert response.status_code == 201
    assert predictor.seen == ["Laptop will not connect to Wi-Fi."]
    assert not hasattr(predictor, "fit")


def test_real_api_uses_saved_pipelines_without_fitting(tmp_path):
    service = create_application_service(tmp_path / "saved_pipeline_api.sqlite3")
    models = [service.prediction_service._department_model,
              service.prediction_service._priority_model]
    for model in models:
        model.fit = Mock(side_effect=AssertionError("inference must not fit"))
    api = create_api_app(service)

    response = _post(api, {
        "body": "The office VPN keeps disconnecting and I cannot access the portal."
    })

    assert response.status_code == 201
    assert response.json()["predicted_department"]
    assert response.json()["predicted_priority"] in {"high", "medium", "low"}
    assert service.database.get_aggregate_counts()["total_tickets"] == 1
    for model in models:
        model.fit.assert_not_called()
