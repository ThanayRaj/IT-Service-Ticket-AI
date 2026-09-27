from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from src.application_service import TicketApplicationService
from src.database import TicketDatabase
from src.demo_tickets import DEMO_TICKET_BODIES, clear_demo_tickets, load_demo_tickets
from src.prediction import PredictionService
from src.sla import SLARiskEngine


@pytest.fixture
def service(tmp_path):
    return TicketApplicationService(
        PredictionService(), TicketDatabase(tmp_path / "demo.sqlite3"), SLARiskEngine()
    )


def test_demo_ticket_texts_are_unlabeled_samples():
    assert 8 <= len(DEMO_TICKET_BODIES) <= 15
    assert all(isinstance(body, str) and body.strip() for body in DEMO_TICKET_BODIES)


def test_demo_tickets_run_saved_models_sla_and_database_without_fitting(service):
    models = [service.prediction_service._department_model,
              service.prediction_service._priority_model]
    for model in models:
        model.fit = Mock(side_effect=AssertionError("demo inference must not fit models"))

    result = load_demo_tickets(service)
    tickets = service.database.get_tickets(source="demo")

    assert result == {"created": len(DEMO_TICKET_BODIES), "skipped": 0,
                      "total": len(DEMO_TICKET_BODIES)}
    assert len(tickets) == len(DEMO_TICKET_BODIES)
    assert all(row["source"] == "demo" for row in tickets)
    assert all(row["source_key"].startswith("demo-ticket-") for row in tickets)
    assert {row["predicted_department"] for row in tickets}
    assert all(row["predicted_priority"] in {"high", "medium", "low"} for row in tickets)
    assert len(service.get_ticket_history()) == len(DEMO_TICKET_BODIES)
    assert service.get_analytics()["source_counts"]["demo"] == len(DEMO_TICKET_BODIES)


def test_repeat_load_is_idempotent_and_clear_preserves_non_demo_records(service):
    manual = service.submit_ticket("User submitted ticket", source="incoming")["ticket"]
    first = load_demo_tickets(service)
    second = load_demo_tickets(service)

    assert first["created"] == len(DEMO_TICKET_BODIES)
    assert second == {"created": 0, "skipped": len(DEMO_TICKET_BODIES),
                      "total": len(DEMO_TICKET_BODIES)}
    assert service.database.count_tickets_by_source("demo") == len(DEMO_TICKET_BODIES)
    assert clear_demo_tickets(service.database) == len(DEMO_TICKET_BODIES)
    assert service.database.count_tickets_by_source("demo") == 0
    assert service.database.get_ticket_by_id(manual["id"])["source"] == "incoming"


def test_streamlit_does_not_autoload_then_loads_and_clears_demo_data(tmp_path, monkeypatch):
    database_path = tmp_path / "streamlit-demo.sqlite3"
    monkeypatch.setenv("IT_SUPPORT_AI_DATABASE_PATH", str(database_path))
    app = AppTest.from_file("../app/app.py").run(timeout=45)
    database = TicketDatabase(database_path)
    existing = database.insert_ticket(
        "User-created record", "IT Support", "medium", source="api"
    )
    app.run(timeout=45)

    assert not app.exception
    assert database.count_tickets_by_source("demo") == 0
    assert "These are not real customer complaints." in " ".join(x.value for x in app.caption)
    assert any("Training CSV data is used only for model development" in x.value
               for x in app.caption)

    app.button(key="load_demo_tickets").click().run(timeout=60)
    assert not app.exception
    assert database.count_tickets_by_source("demo") == len(DEMO_TICKET_BODIES)
    assert app.button(key="load_demo_tickets").disabled
    assert any("Loaded" in x.value for x in app.success)

    assert any("Total Tickets" in x.value for x in app.markdown)
    app.sidebar.radio(key="nav_operations").set_value("Ticket History").run(timeout=45)
    assert not app.exception
    assert any("Ticket history" in x.value for x in app.markdown)
    assert any("Demo" in x.value for x in app.caption)
    app.sidebar.radio(key="nav_operations").set_value("Analytics").run(timeout=45)
    assert not app.exception
    assert any("Ticket analytics" in x.value for x in app.markdown)
    assert any("Ticket origins" in x.value for x in app.markdown)

    app.sidebar.radio(key="nav_overview").set_value("Bulk Processing").run(timeout=45)
    assert not app.exception
    assert any("Process tickets in bulk" in x.value for x in app.markdown)
    app.sidebar.radio(key="nav_system").set_value("AI Models").run(timeout=45)
    assert not app.exception
    assert any("AI models" in x.value for x in app.markdown)
    app.sidebar.radio(key="nav_overview").set_value("Incoming Tickets").run(timeout=45)
    assert not app.exception
    assert any("Ticket intake" in x.value for x in app.markdown)

    app.sidebar.radio(key="nav_overview").set_value("Overview").run(timeout=45)
    app.button(key="clear_demo_data").click().run(timeout=45)
    assert any("This removes only records marked as demo data." in item.value
               for item in app.warning)
    app.button(key="confirm_demo_clear").click().run(timeout=45)
    assert not app.exception
    assert database.count_tickets_by_source("demo") == 0
    assert database.get_ticket_by_id(existing["id"])["source"] == "api"
    assert database.get_aggregate_counts()["total_tickets"] == 1
    assert any("Removed" in x.value for x in app.success)
    app.sidebar.radio(key="nav_operations").set_value("Ticket History").run(timeout=45)
    assert not app.exception
    assert any("User-created record" in x.value for x in app.markdown)
    assert not any("VPN / internal" in x.value for x in app.markdown)
    app.sidebar.radio(key="nav_operations").set_value("Analytics").run(timeout=45)
    assert not app.exception
    assert any("Ticket origins" in x.value for x in app.markdown)
    assert any("Tickets processed" in x.value for x in app.markdown)


def test_streamlit_manual_intake_uses_the_normal_pipeline(tmp_path, monkeypatch):
    database_path = tmp_path / "manual-intake.sqlite3"
    monkeypatch.setenv("IT_SUPPORT_AI_DATABASE_PATH", str(database_path))
    app = AppTest.from_file("../app/app.py").run(timeout=45)
    app.sidebar.radio(key="nav_overview").set_value("Incoming Tickets").run(timeout=45)
    body = "The VPN will not connect and I cannot access the internal support portal."
    app.text_area[0].set_value(body)
    app.button(key="FormSubmitter:ticket_intake_form-Process ticket").click().run(timeout=60)

    database = TicketDatabase(database_path)
    tickets = database.get_tickets()
    assert not app.exception
    assert len(tickets) == 1
    assert tickets[0]["body"] == body
    assert tickets[0]["source"] == "incoming"
    assert tickets[0]["predicted_department"]
    assert tickets[0]["predicted_priority"] in {"high", "medium", "low"}
    assert any("Intake workflow" in item.value for item in app.markdown)
