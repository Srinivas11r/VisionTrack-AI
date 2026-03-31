import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import configure_logging
from core.processor import VideoProcessor
from core.settings import load_settings
from database.models import Database
from utils.csv_export import export_summary_csv, export_tracks_to_csv

configure_logging()
logger = logging.getLogger("app.api")


BASE_DIR = Path(__file__).parent.parent.parent
settings = load_settings(BASE_DIR)

app = FastAPI(title=settings.app_title)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
processor: VideoProcessor | None = None
processor_lock = Lock()
processing_jobs = {}
jobs_lock = Lock()

# Directories
UPLOAD_DIR = settings.uploads_dir
OUTPUT_DIR = settings.outputs_dir
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

db = Database(settings.db_url)


def get_processor() -> VideoProcessor:
    global processor
    with processor_lock:
        if processor is None:
            processor = VideoProcessor(
                model_name=settings.model_name,
                performance_mode=settings.default_performance_mode,
                ocr_enabled=settings.ocr_enabled,
                max_objects_per_frame=settings.max_objects_per_frame,
            )
    return processor


def set_processor(performance_mode: bool) -> VideoProcessor:
    global processor
    with processor_lock:
        processor = VideoProcessor(
            model_name=settings.model_name,
            performance_mode=performance_mode,
            ocr_enabled=settings.ocr_enabled,
            max_objects_per_frame=settings.max_objects_per_frame,
        )
    return processor


class Specifications(BaseModel):
    object_type: str = "all"
    color: str = "all"
    size: str = "all"
    confidence: float = 0.5


class ProcessingMode(BaseModel):
    performance_mode: bool = False  # True = Fast (no OCR), False = Full features


@app.get("/")
def read_root():
    return {
        "message": "Object Detection & Tracking API",
        "version": "1.0",
        "endpoints": ["/upload-video", "/set-specifications", "/logs", "/clear-logs", "/reset"],
    }


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    PART 1 & 8: Upload and process video with comprehensive error handling
    """
    filename = file.filename or "uploaded_video.mp4"
    if not filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file format. Use mp4, avi, or mov")

    # Save uploaded file
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("Video uploaded: %s", filename)

    # Process video
    output_path = OUTPUT_DIR / f"processed_{filename}"

    try:
        result = get_processor().process_video(str(file_path), str(output_path))

        # Check for errors
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Extract data from new return format
        logs = result.get("logs", [])
        metadata = result.get("metadata", {})
        warnings = result.get("warnings", [])
        stats = result.get("stats", {})

        # Save logs to database
        for log in logs:
            db.add_track_log(log)

        return JSONResponse(
            {
                "status": "success",
                "message": "Video processed successfully",
                "output_file": str(output_path),
                "tracked_objects": len(logs),
                "logs": logs,
                "metadata": {
                    "duration": f"{metadata.get('duration', 0):.2f} seconds",
                    "fps": f"{metadata.get('fps', 0):.2f}",
                    "resolution": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                    "total_frames": metadata.get("total_frames", 0),
                },
                "stats": {
                    "frames_processed": stats.get("frames_processed", 0),
                    "processing_time": f"{stats.get('processing_time', 0):.2f} seconds",
                    "avg_fps": f"{stats.get('avg_fps', 0):.2f}",
                    "unique_objects": stats.get("unique_objects", 0),
                },
                "warnings": warnings,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload video processing failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


def _process_video_job(job_id: str, file_path: Path, output_path: Path):
    """Background job runner for async video processing with progress updates."""
    try:
        with jobs_lock:
            processing_jobs[job_id].update(
                {"status": "processing", "message": "Starting processing...", "progress": 0}
            )

        def progress_callback(progress_data):
            with jobs_lock:
                if job_id in processing_jobs:
                    processing_jobs[job_id].update(
                        {
                            "progress": progress_data.get("progress", 0),
                            "current_frame": progress_data.get("current_frame", 0),
                            "total_frames": progress_data.get("total_frames", 0),
                            "message": progress_data.get("message", "Processing..."),
                        }
                    )

        result = get_processor().process_video(
            str(file_path), str(output_path), progress_callback=progress_callback
        )

        if isinstance(result, dict) and "error" in result:
            with jobs_lock:
                processing_jobs[job_id].update(
                    {"status": "error", "message": result["error"], "progress": 100}
                )
            return

        logs = result.get("logs", [])
        metadata = result.get("metadata", {})
        warnings = result.get("warnings", [])
        stats = result.get("stats", {})

        for log in logs:
            db.add_track_log(log)

        final_result = {
            "status": "success",
            "message": "Video processed successfully",
            "output_file": str(output_path),
            "tracked_objects": len(logs),
            "logs": logs,
            "metadata": {
                "duration": f"{metadata.get('duration', 0):.2f} seconds",
                "fps": f"{metadata.get('fps', 0):.2f}",
                "resolution": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                "total_frames": metadata.get("total_frames", 0),
            },
            "stats": {
                "frames_processed": stats.get("frames_processed", 0),
                "processing_time": f"{stats.get('processing_time', 0):.2f} seconds",
                "avg_fps": f"{stats.get('avg_fps', 0):.2f}",
                "unique_objects": stats.get("unique_objects", 0),
            },
            "warnings": warnings,
        }

        with jobs_lock:
            processing_jobs[job_id].update(
                {
                    "status": "completed",
                    "message": "Processing complete",
                    "progress": 100,
                    "result": final_result,
                }
            )

    except Exception as e:
        logger.exception("Background processing failed for job_id=%s", job_id)
        with jobs_lock:
            if job_id in processing_jobs:
                processing_jobs[job_id].update(
                    {"status": "error", "message": f"Processing error: {str(e)}", "progress": 100}
                )


@app.post("/upload-video-async")
async def upload_video_async(file: UploadFile = File(...)):
    """Upload video and start async processing job with progress reporting."""
    filename = file.filename or "uploaded_video.mp4"
    if not filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file format. Use mp4, avi, or mov")

    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = OUTPUT_DIR / f"processed_{filename}"
    job_id = str(uuid.uuid4())

    with jobs_lock:
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "message": "Job queued",
            "filename": filename,
        }

    worker = Thread(target=_process_video_job, args=(job_id, file_path, output_path), daemon=True)
    worker.start()

    return JSONResponse(
        {
            "status": "accepted",
            "job_id": job_id,
            "message": "Video upload successful. Processing started.",
        }
    )


@app.get("/processing-status/{job_id}")
def get_processing_status(job_id: str):
    with jobs_lock:
        job = processing_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "current_frame": job.get("current_frame", 0),
        "total_frames": job.get("total_frames", 0),
        "message": job.get("message", ""),
    }

    if job.get("status") == "completed" and job.get("result"):
        response.update(
            {
                "tracked_objects": job["result"].get("tracked_objects", 0),
                "output_file": job["result"].get("output_file", ""),
            }
        )

    return JSONResponse(response)


@app.get("/processing-result/{job_id}")
def get_processing_result(job_id: str):
    with jobs_lock:
        job = processing_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") == "error":
        raise HTTPException(status_code=500, detail=job.get("message", "Processing failed"))

    if job.get("status") != "completed":
        return JSONResponse(
            {"status": "processing", "message": "Result not ready yet"}, status_code=202
        )

    return JSONResponse(job.get("result", {}))


@app.post("/set-specifications")
def set_specifications(specs: Specifications):
    """
    Set filtering specifications
    """
    active_processor = get_processor()
    active_processor.set_specifications(specs.dict())

    return JSONResponse({"status": "success", "specifications": specs.dict()})


@app.get("/specifications")
def get_specifications():
    """
    Get current specifications
    """
    return JSONResponse({"specifications": get_processor().specifications})


@app.get("/logs")
def get_logs(limit: int = 100):
    """
    Get tracking logs from database
    """
    logs = db.get_all_logs(limit=limit)
    return JSONResponse({"count": len(logs), "logs": logs})


@app.delete("/clear-logs")
def clear_logs():
    """
    Clear all logs from database
    """
    db.clear_logs()
    return JSONResponse({"status": "success", "message": "All logs cleared"})


@app.post("/reset")
def reset_processor():
    """
    Reset processor state
    """
    get_processor().reset()
    return JSONResponse({"status": "success", "message": "Processor reset"})


@app.get("/download/{filename}")
def download_file(filename: str):
    """
    Download processed video or CSV files
    """
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path), filename=filename)


@app.post("/set-processing-mode")
def set_processing_mode(mode: ProcessingMode):
    """
    Switch between performance mode (fast) and full feature mode
    - Performance mode: ~3-4x faster, no license plate OCR, frame skipping
    - Full mode: Slower, complete feature set including OCR
    """
    try:
        set_processor(performance_mode=mode.performance_mode)
        return JSONResponse(
            {
                "status": "success",
                "mode": "PERFORMANCE (Fast)" if mode.performance_mode else "FULL FEATURES (Slow)",
                "performance_mode": mode.performance_mode,
                "features": {
                    "ocr_enabled": not mode.performance_mode,
                    "frame_skip": 3 if mode.performance_mode else 1,
                    "expected_speedup": "~3-4x faster" if mode.performance_mode else "Standard",
                },
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting mode: {str(e)}")


@app.post("/export-csv")
def export_csv():
    """
    Export tracking results to CSV with detailed attributes
    """
    try:
        # Generate timestamp for unique filename
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        active_processor = get_processor()

        # Export detailed CSV
        csv_path = f"storage/outputs/tracking_results_{timestamp}.csv"
        result_path = export_tracks_to_csv(active_processor.tracker, output_path=csv_path)

        # Export summary CSV
        summary_path = f"storage/outputs/tracking_summary_{timestamp}.csv"
        summary_result = export_summary_csv(active_processor.tracker, output_path=summary_path)

        if result_path and summary_result:
            return JSONResponse(
                {
                    "status": "success",
                    "message": "CSV files exported successfully",
                    "detailed_csv": result_path,
                    "summary_csv": summary_result,
                    "total_tracks": len(active_processor.tracker.get_all_tracks_with_attributes()),
                }
            )
        else:
            raise HTTPException(status_code=400, detail="No tracking data to export")

    except Exception as e:
        logger.exception("CSV export failed")
        raise HTTPException(status_code=500, detail=f"CSV export error: {str(e)}")


@app.get("/download-csv/{filename}")
def download_csv(filename: str):
    """
    Download CSV file
    """
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    return FileResponse(str(file_path), filename=filename, media_type="text/csv")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
