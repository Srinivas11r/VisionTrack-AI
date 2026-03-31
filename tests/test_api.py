import io

import pytest
from fastapi.testclient import TestClient

import backend.api.main as api_main
from backend.api.main import app, jobs_lock, processing_jobs

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_processing_jobs():
    with jobs_lock:
        processing_jobs.clear()
    yield
    with jobs_lock:
        processing_jobs.clear()


def test_health_endpoint_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint_has_expected_shape():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "message" in payload
    assert "version" in payload
    assert "endpoints" in payload


def test_processing_status_unknown_job_returns_404():
    response = client.get("/processing-status/non-existent-job")
    assert response.status_code == 404


def test_processing_result_unknown_job_returns_404():
    response = client.get("/processing-result/non-existent-job")
    assert response.status_code == 404


def test_processing_result_returns_202_when_not_completed():
    job_id = "test-job-processing"
    with jobs_lock:
        processing_jobs[job_id] = {"status": "processing", "message": "Working"}

    response = client.get(f"/processing-result/{job_id}")
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "processing"


def test_upload_video_rejects_invalid_extension():
    payload = io.BytesIO(b"not-a-video")
    files = {"file": ("bad.txt", payload, "text/plain")}
    response = client.post("/upload-video", files=files)
    assert response.status_code == 400


def test_upload_video_success_with_mocked_processor(monkeypatch, tmp_path):
    class DummyProcessor:
        def process_video(self, _input_path, _output_path):
            return {
                "logs": [{"Object_ID": 1, "Type": "person", "Attributes": "Unknown"}],
                "metadata": {"duration": 1.2, "fps": 30.0, "width": 1280, "height": 720},
                "warnings": [],
                "stats": {"frames_processed": 30, "processing_time": 1.0, "avg_fps": 30.0},
            }

    monkeypatch.setattr(api_main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_main, "OUTPUT_DIR", tmp_path / "outputs")
    api_main.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    api_main.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "get_processor", lambda: DummyProcessor())
    monkeypatch.setattr(api_main.db, "add_track_log", lambda _log: None)

    payload = io.BytesIO(b"fake-video-content")
    files = {"file": ("clip.mp4", payload, "video/mp4")}
    response = client.post("/upload-video", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["tracked_objects"] == 1


def test_processing_result_returns_500_for_error_job():
    job_id = "test-job-error"
    with jobs_lock:
        processing_jobs[job_id] = {"status": "error", "message": "boom"}

    response = client.get(f"/processing-result/{job_id}")
    assert response.status_code == 500


def test_processing_status_includes_summary_for_completed_job():
    job_id = "test-job-done"
    with jobs_lock:
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "current_frame": 120,
            "total_frames": 120,
            "message": "done",
            "result": {"tracked_objects": 3, "output_file": "outputs/processed_clip.mp4"},
        }

    response = client.get(f"/processing-status/{job_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["tracked_objects"] == 3
    assert payload["output_file"] == "outputs/processed_clip.mp4"


def test_upload_video_async_starts_job_and_returns_job_id(monkeypatch, tmp_path):
    class InlineThread:
        def __init__(self, target, args=(), daemon=False):
            self._target = target
            self._args = args
            self.daemon = daemon

        def start(self):
            self._target(*self._args)

    def fake_process_video_job(job_id, _file_path, _output_path):
        with jobs_lock:
            processing_jobs[job_id].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "message": "Processing complete",
                    "result": {
                        "status": "success",
                        "tracked_objects": 1,
                        "output_file": "outputs/processed_clip.mp4",
                    },
                }
            )

    monkeypatch.setattr(api_main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_main, "OUTPUT_DIR", tmp_path / "outputs")
    api_main.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    api_main.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "Thread", InlineThread)
    monkeypatch.setattr(api_main, "_process_video_job", fake_process_video_job)

    payload = io.BytesIO(b"fake-video-content")
    files = {"file": ("clip.mp4", payload, "video/mp4")}
    response = client.post("/upload-video-async", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert "job_id" in body

    status_response = client.get(f"/processing-status/{body['job_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"


def test_logs_endpoint_schema(monkeypatch):
    sample_logs = [
        {
            "Object_ID": 1,
            "Type": "person",
            "Attributes": "Upper: Blue",
            "First_Seen": "2026-01-01T00:00:00",
            "Last_Seen": "2026-01-01T00:00:01",
            "Duration": 1.0,
        }
    ]
    monkeypatch.setattr(api_main.db, "get_all_logs", lambda limit=100: sample_logs[:limit])

    response = client.get("/logs")
    assert response.status_code == 200
    payload = response.json()
    assert "count" in payload
    assert "logs" in payload
    assert isinstance(payload["logs"], list)
    assert set(["Object_ID", "Type", "Attributes", "First_Seen", "Last_Seen", "Duration"]).issubset(
        payload["logs"][0].keys()
    )


def test_export_csv_contract_schema(monkeypatch):
    class DummyTracker:
        def get_all_tracks_with_attributes(self):
            return [{"id": 1}]

    class DummyProcessor:
        tracker = DummyTracker()

    monkeypatch.setattr(api_main, "get_processor", lambda: DummyProcessor())
    monkeypatch.setattr(
        api_main,
        "export_tracks_to_csv",
        lambda tracker, output_path=None: output_path or "outputs/tracking_results.csv",
    )
    monkeypatch.setattr(
        api_main,
        "export_summary_csv",
        lambda tracker, output_path=None: output_path or "outputs/tracking_summary.csv",
    )

    response = client.post("/export-csv")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "detailed_csv" in payload
    assert "summary_csv" in payload
    assert "total_tracks" in payload
