import pandas as pd
import pytest

from src.bulk_processing import BulkInputError, process_bulk_frame, validate_bulk_frame
from src.service_factory import create_application_service


class RecordingApplicationService:
    def __init__(self):
        self.bodies = []
        self.sources = []

    def submit_ticket(self, body, source="incoming", **kwargs):
        self.bodies.append(body)
        self.sources.append(source)
        n = len(self.bodies)
        return {
            "prediction": {"department": "IT Support", "priority": "medium"},
            "sla": {"status": "Within SLA"},
            "ticket": {"id": n, "created_at": "2026-09-26T08:00:00.000Z"},
        }


def test_body_column_is_required():
    with pytest.raises(BulkInputError, match="Body"):
        validate_bulk_frame(pd.DataFrame({"description": ["A ticket"]}))


@pytest.mark.parametrize("body", [123, 42.5, True])
def test_nontext_nonblank_body_is_rejected(body):
    with pytest.raises(BulkInputError, match="must be text"):
        validate_bulk_frame(pd.DataFrame({"Body": [body]}))


def test_blank_rows_are_preserved_but_not_processed_or_stored():
    source = pd.DataFrame({"Body": ["A laptop cannot connect.", "", "   "], "Source": ["a", "b", "c"]})
    service = RecordingApplicationService()
    result = process_bulk_frame(source, service)

    assert list(result["Source"]) == ["a", "b", "c"]
    assert len(result) == 3
    assert result.loc[0, "predicted_department"] == "IT Support"
    assert result.loc[0, "predicted_priority"] == "medium"
    assert result.loc[0, "sla_status"] == "Within SLA"
    assert result.loc[0, "application_created_at"].endswith("Z")
    assert result.loc[1:, "predicted_department"].tolist() == ["", ""]
    assert service.bodies == ["A laptop cannot connect."]
    assert service.sources == ["bulk"]


def test_uploaded_target_columns_are_metadata_and_body_is_only_input():
    source = pd.DataFrame({
        "Body": ["A laptop cannot connect."],
        "Department": ["forged input label"],
        "Priority": ["low"],
        "Tags": ["target leakage risk"],
    })
    service = RecordingApplicationService()
    result = process_bulk_frame(source, service)

    assert service.bodies == ["A laptop cannot connect."]
    assert result.loc[0, "Department"] == "forged input label"
    assert result.loc[0, "Priority"] == "low"
    assert result.loc[0, "predicted_department"] == "IT Support"
    assert result.loc[0, "predicted_priority"] == "medium"
    assert service.sources == ["bulk"]


@pytest.mark.parametrize("reserved", ["predicted_department", "predicted_priority", "sla_status",
                                       "application_created_at"])
def test_reserved_output_columns_are_rejected_to_prevent_silent_overwrite(reserved):
    with pytest.raises(BulkInputError, match="reserved output"):
        validate_bulk_frame(pd.DataFrame({"Body": ["text"], reserved: ["existing"]}))


def test_progress_callback_reports_completion_and_no_rows():
    source = pd.DataFrame({"Body": ["one", "two"]})
    progress = []
    process_bulk_frame(source, RecordingApplicationService(), progress.append)
    assert progress == [0.0, 0.5, 1.0]

    progress = []
    process_bulk_frame(pd.DataFrame({"Body": [""]}), RecordingApplicationService(), progress.append)
    assert progress == [0.0]


def test_real_bulk_inference_persists_predictions_without_fitting(tmp_path):
    service = create_application_service(tmp_path / "bulk_inference.sqlite3")
    models = [service.prediction_service._department_model,
              service.prediction_service._priority_model]
    for model in models:
        model.fit = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk inference must not fit")
        )
    source = pd.DataFrame({
        "Body": ["The office VPN keeps disconnecting and blocks portal access.", ""],
        "Department": ["metadata only", ""],
        "Priority": ["metadata only", ""],
    })

    result = process_bulk_frame(source, service)

    assert len(result) == 2
    assert result.loc[0, "predicted_department"]
    assert result.loc[0, "predicted_priority"] in {"high", "medium", "low"}
    assert result.loc[0, "sla_status"] == "Within SLA"
    assert result.loc[1, "predicted_department"] == ""
    assert result.loc[1, "Department"] == ""
    assert service.database.get_aggregate_counts()["total_tickets"] == 1
    assert service.database.get_tickets()[0]["source"] == "bulk"


def test_processed_batch_can_be_reconstructed_after_service_reload(tmp_path):
    database_path = tmp_path / "bulk_reload.sqlite3"
    service = create_application_service(database_path)
    batch_id = "stable-upload-id"
    source = pd.DataFrame({"Body": ["A user cannot access email."], "Customer": ["Example Co"]})
    result = process_bulk_frame(source, service, batch_id=batch_id)

    reopened = create_application_service(database_path)
    from src.bulk_processing import bulk_frame_from_records
    restored = bulk_frame_from_records(reopened.database.get_bulk_batch(batch_id))

    assert reopened.database.get_latest_bulk_batch() == batch_id
    assert restored.loc[0, "Body"] == result.loc[0, "Body"]
    assert restored.loc[0, "Customer"] == "Example Co"
    assert restored.loc[0, "predicted_department"] == result.loc[0, "predicted_department"]
    assert reopened.database.get_bulk_batch(batch_id)[0]["source"] == "bulk"
