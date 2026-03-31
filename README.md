# Intelligent Specification-Based Object Detection and Tracking System

## Project Overview
Production-oriented video object analytics system using YOLOv8, FastAPI, and Streamlit.

The platform detects and tracks objects, extracts dynamic attributes, and provides async processing with progress polling and CSV export.

## Architecture Diagram
See [docs/architecture.md](docs/architecture.md) for the full flow diagram.

## Features
- YOLOv8-based multi-class object detection.
- IoU-based object identity tracking across frames.
- Filter pipeline: object type, color, size, confidence.
- Attribute extraction by object family (person, vehicle, bag, animal, generic).
- Async processing pipeline with status and result polling.
- CSV export and logs endpoints.
- Environment-based runtime settings for dev/prod.
- Docker and Docker Compose support.

## Quick Start
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env

# backend
python -m uvicorn backend.api.main:app --reload --port 8000

# frontend (new terminal)
streamlit run frontend/app.py
```

## API Endpoints
- `POST /upload-video`
- `POST /upload-video-async`
- `GET /processing-status/{job_id}`
- `GET /processing-result/{job_id}`
- `POST /set-specifications`
- `POST /set-processing-mode`
- `GET /logs`
- `POST /export-csv`
- `GET /download/{filename}`
- `GET /download-csv/{filename}`
- `GET /health`

## Data Model and CSV Format
Minimal tracking schema:
- `Object_ID`
- `Type`
- `Attributes`
- `First_Seen`
- `Last_Seen`
- `Duration`

Logs and CSV exports are generated from tracked object state and normalized attribute mappings.

## Performance Mode Details
- `PERFORMANCE_MODE=true` or `DEFAULT_PERFORMANCE_MODE=true` enables faster processing defaults.
- `OCR_ENABLED=false` disables OCR extraction overhead.
- `MAX_OBJECTS_PER_FRAME` caps frame-level workload in dense scenes.

## Limitations and Roadmap
Current limitations:
- Default in-process async jobs are single-service memory scoped.
- Database auto-migration helper is sqlite-focused.

Planned work is maintained in [docs/roadmap.md](docs/roadmap.md).

## Contribution and License
- Run quality checks before PRs:
  - `black .`
  - `isort .`
  - `ruff check .`
  - `pytest`
- CI runs lint, tests, and Docker image build smoke checks.

License: MIT.
