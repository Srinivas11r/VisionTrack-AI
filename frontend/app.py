import os
import tempfile
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AI Object Analytics Dashboard", page_icon="🎯", layout="wide")

API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("STREAMLIT_API_URL", API_URL)

OBJECT_CLASSES = [
    "all",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "bed",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "clock",
    "vase",
]
COLORS = ["all", "red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
SIZES = ["all", "small", "medium", "large"]

DISPLAY_COLUMNS = [
    "track_id",
    "class",
    "first_seen",
    "last_seen",
    "duration",
    "frame_count",
    "gender",
    "age_group",
    "shirt_color",
    "pant_color",
    "number_plate",
    "vehicle_color",
    "vehicle_company",
    "body_type",
    "bag_color",
    "bag_type",
    "animal_species",
    "object_color",
]


def apply_dashboard_styles():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            max-width: 1400px;
        }
        .status-card {
            padding: 10px 14px;
            border: 1px solid #2f2f2f;
            border-radius: 12px;
            background: #111214;
            min-height: 68px;
        }
        .status-label {
            font-size: 12px;
            color: #9aa0a6;
            line-height: 1.2;
        }
        .status-value {
            font-size: 14px;
            font-weight: 700;
            margin-top: 3px;
            line-height: 1.2;
        }
        .panel-title {
            font-size: 15px;
            font-weight: 700;
            margin: 2px 0 8px 0;
        }
        .helper-tip {
            font-size: 12px;
            color: #9aa0a6;
            margin-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "alerts": [],
        "processing_result": None,
        "processing_metrics": {
            "frames_processed": 0,
            "active_objects": 0,
            "unique_ids": 0,
            "fps": 0.0,
        },
        "video_info": {},
        "last_uploaded_name": None,
        "current_job_id": None,
        "input_source": "Upload Video",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_alert(level, message):
    st.session_state.alerts.append((level, message))
    st.session_state.alerts = st.session_state.alerts[-8:]


def check_backend_running():
    try:
        response = requests.get(f"{API_URL}/health", timeout=4)
        return response.status_code == 200
    except Exception:
        return False


def status_badge(label, value, color):
    st.markdown(
        f"<div class='status-card'>"
        f"<div class='status-label'>{label}</div>"
        f"<div class='status-value' style='color:{color};'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_header(backend_ok, input_source):
    st.title("AI Object Analytics Dashboard")
    st.caption("Real-time detection using YOLOv8")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_status = "Loaded" if backend_ok else "Not Loaded"
        model_color = "#22c55e" if backend_ok else "#ef4444"
        status_badge("Model Status", model_status, model_color)
    with col2:
        backend_status = "Running" if backend_ok else "Offline"
        backend_color = "#22c55e" if backend_ok else "#ef4444"
        status_badge("Backend Status", backend_status, backend_color)
    with col3:
        source_color = "#eab308" if input_source == "Use Webcam" else "#22c55e"
        status_badge("Input Source", input_source.replace("Use ", ""), source_color)


def render_sidebar(backend_ok):
    device = "CPU"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "GPU"
    except Exception:
        pass

    st.sidebar.header("System Information")
    st.sidebar.write("**Model:** yolov8n")
    st.sidebar.write(f"**Device:** {device}")
    st.sidebar.write("**Version:** 1.0")
    st.sidebar.write("**Developer:** Intelligent Analytics Team")
    st.sidebar.write("**Backend:** " + ("Running" if backend_ok else "Offline"))
    st.sidebar.markdown("---")
    st.sidebar.subheader("Future Features")
    st.sidebar.write("- Multi-camera monitoring")
    st.sidebar.write("- Cross-camera re-identification")
    st.sidebar.write("- Alert rules engine")
    st.sidebar.write("- Heatmap analytics")


def render_controls(left_col):
    with left_col:
        st.markdown("<div class='panel-title'>🎛 Controls</div>", unsafe_allow_html=True)
        st.markdown("#### Input Source")
        input_source = st.radio(
            "Input Source",
            ["Upload Video", "Use Webcam"],
            index=0 if st.session_state.input_source == "Upload Video" else 1,
            label_visibility="collapsed",
        )
        st.session_state.input_source = input_source

        uploaded_file = None
        if input_source == "Upload Video":
            uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        else:
            st.info("Webcam mode selected. Live preview support is shown in output panel.")

        st.markdown("#### Detection Filters")
        object_type = st.selectbox("Object Type", OBJECT_CLASSES, index=0)
        color = st.selectbox("Color", COLORS, index=0)
        size = st.selectbox("Size", SIZES, index=0)
        confidence = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05)

        st.markdown("#### Processing Controls")
        c1, c2, c3 = st.columns(3)
        start_disabled = input_source == "Upload Video" and uploaded_file is None
        start_clicked = c1.button("Start", type="primary", disabled=start_disabled)
        stop_clicked = c2.button("Stop")
        reset_clicked = c3.button("Reset")

        st.markdown(
            "<div class='helper-tip'>Tip: Use All filters for initial testing</div>",
            unsafe_allow_html=True,
        )

        return {
            "input_source": input_source,
            "uploaded_file": uploaded_file,
            "filters": {
                "object_type": object_type,
                "color": color,
                "size": size,
                "confidence": confidence,
            },
            "start_clicked": start_clicked,
            "stop_clicked": stop_clicked,
            "reset_clicked": reset_clicked,
        }


def render_video_panel(right_col, input_source):
    with right_col:
        st.markdown("<div class='panel-title'>📹 Live Detection</div>", unsafe_allow_html=True)
        video_placeholder = st.empty()
        progress_placeholder = st.empty()
        frame_placeholder = st.empty()

        result = st.session_state.processing_result
        if result and result.get("output_file"):
            video_placeholder.video(result["output_file"])
        elif input_source == "Upload Video":
            video_placeholder.info("Upload a video and click Start")
        else:
            video_placeholder.info("Webcam preview will appear here when available.")

        return video_placeholder, progress_placeholder, frame_placeholder


def render_metrics(right_col):
    with right_col:
        st.markdown("<div class='panel-title'>📊 Statistics</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        metrics = st.session_state.processing_metrics
        m1.metric("Active Objects", int(metrics.get("active_objects", 0)))
        m2.metric("Total Unique Objects", int(metrics.get("unique_ids", 0)))
        m3.metric("Frames Processed", int(metrics.get("frames_processed", 0)))
        m4.metric("Processing FPS", f"{metrics.get('fps', 0.0):.2f}")


def render_logs(logs):
    with st.expander("🧾 Logs - Tracking Logs", expanded=True):
        if not logs:
            st.info("No tracking logs available yet.")
            return

        df = pd.DataFrame(logs)
        base_cols = ["Object_ID", "Type", "Attributes", "First_Seen", "Last_Seen", "Duration"]
        cols = [c for c in base_cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)


def render_alerts():
    st.markdown("<div class='panel-title'>🚨 Events / Alerts</div>", unsafe_allow_html=True)
    if not st.session_state.alerts:
        st.info("No alerts yet.")
        return

    for level, message in st.session_state.alerts[-5:]:
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.error(message)


def render_video_info():
    st.markdown("<div class='panel-title'>📁 Video Info</div>", unsafe_allow_html=True)
    info = st.session_state.video_info
    if not info:
        st.info("Video metadata will appear after processing starts.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filename", info.get("filename", "N/A"))
    c2.metric("Duration", info.get("duration", "N/A"))
    c3.metric("FPS", info.get("fps", "N/A"))
    c4.metric("Resolution", info.get("resolution", "N/A"))
    c5.metric("Frame Count", info.get("total_frames", "N/A"))


def render_downloads():
    st.markdown("<div class='panel-title'>⬇️ Downloads</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Download CSV"):
            try:
                export_resp = requests.post(f"{API_URL}/export-csv", timeout=20)
                if export_resp.status_code != 200:
                    add_alert("error", f"CSV export failed: {export_resp.text}")
                    return

                data = export_resp.json()
                csv_path = data.get("detailed_csv", "")
                candidate_paths = [
                    csv_path,
                    os.path.join("backend", csv_path) if csv_path else "",
                    (
                        os.path.join("backend", "outputs", os.path.basename(csv_path))
                        if csv_path
                        else ""
                    ),
                ]

                found = None
                for path in candidate_paths:
                    if path and os.path.exists(path):
                        found = path
                        break

                if found:
                    with open(found, "rb") as fh:
                        st.download_button(
                            label="Download CSV File",
                            data=fh.read(),
                            file_name=os.path.basename(found),
                            mime="text/csv",
                        )
                    add_alert("success", "CSV ready for download")
                else:
                    add_alert("warning", "CSV exported, but local file path not found")
            except Exception as exc:
                add_alert("error", f"CSV download error: {exc}")

    with c2:
        result = st.session_state.processing_result or {}
        output_path = result.get("output_file")
        if output_path and os.path.exists(output_path):
            with open(output_path, "rb") as vf:
                st.download_button(
                    label="Download Processed Video",
                    data=vf.read(),
                    file_name=os.path.basename(output_path),
                    mime="video/mp4",
                )
        else:
            st.button("Download Processed Video", disabled=True)


def reset_dashboard_state():
    st.session_state.processing_result = None
    st.session_state.video_info = {}
    st.session_state.processing_metrics = {
        "frames_processed": 0,
        "active_objects": 0,
        "unique_ids": 0,
        "fps": 0.0,
    }


def start_processing(uploaded_file, filters, progress_placeholder, frame_placeholder):
    progress = progress_placeholder.progress(0, text="Uploading video...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        requests.post(f"{API_URL}/set-specifications", json=filters, timeout=10)

        with open(tmp_path, "rb") as f:
            files = {"file": (uploaded_file.name, f, "video/mp4")}
            start_response = requests.post(f"{API_URL}/upload-video-async", files=files, timeout=60)

        if start_response.status_code != 200:
            add_alert("error", f"Failed to start processing: {start_response.text}")
            return

        job_id = start_response.json().get("job_id")
        st.session_state.current_job_id = job_id
        add_alert("success", "Processing started")
        add_alert("success", "Video loaded successfully")

        while True:
            status_response = requests.get(f"{API_URL}/processing-status/{job_id}", timeout=30)
            if status_response.status_code != 200:
                add_alert("error", f"Status check failed: {status_response.text}")
                break

            status_data = status_response.json()
            current = int(status_data.get("current_frame", 0))
            total = int(status_data.get("total_frames", 0))
            pct = int(status_data.get("progress", 0))
            state = status_data.get("status", "processing")
            message = status_data.get("message", "Processing...")

            progress.progress(min(max(pct, 0), 100), text=f"{pct}% - {message}")
            frame_placeholder.caption(
                f"Processing frame {current} / {total}" if total > 0 else "Preparing frames..."
            )

            if state == "completed":
                result_response = requests.get(f"{API_URL}/processing-result/{job_id}", timeout=30)
                if result_response.status_code == 200:
                    result = result_response.json()
                    st.session_state.processing_result = result
                    st.session_state.processing_metrics = {
                        "frames_processed": int(result.get("stats", {}).get("frames_processed", 0)),
                        "active_objects": int(result.get("tracked_objects", 0)),
                        "unique_ids": int(result.get("stats", {}).get("unique_objects", 0)),
                        "fps": float(result.get("stats", {}).get("avg_fps", 0)),
                    }
                    st.session_state.video_info = {
                        "filename": uploaded_file.name,
                        "duration": result.get("metadata", {}).get("duration", "N/A"),
                        "fps": result.get("metadata", {}).get("fps", "N/A"),
                        "resolution": result.get("metadata", {}).get("resolution", "N/A"),
                        "total_frames": result.get("metadata", {}).get("total_frames", "N/A"),
                    }
                    if result.get("tracked_objects", 0) == 0:
                        add_alert("warning", "No objects detected")
                    for warning in result.get("warnings", []):
                        add_alert("warning", warning)
                    add_alert("success", "Processing completed")
                    progress.progress(100, text="100% - Processing complete")
                else:
                    add_alert("error", f"Result fetch failed: {result_response.text}")
                break

            if state == "error":
                add_alert("error", status_data.get("message", "Processing failed"))
                break

            time.sleep(1)

    except Exception as exc:
        add_alert("error", f"Processing error: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def fetch_latest_logs(limit=100):
    try:
        response = requests.get(
            f"{API_URL}/logs?limit={limit}", headers={"Cache-Control": "no-cache"}, timeout=20
        )
        if response.status_code == 200:
            return response.json().get("logs", [])
    except Exception:
        return []
    return []


def main():
    init_state()
    apply_dashboard_styles()
    backend_ok = check_backend_running()

    render_header(backend_ok, st.session_state.input_source)
    render_sidebar(backend_ok)

    controls_col, output_col = st.columns([1, 1.6], gap="large")
    controls = render_controls(controls_col)

    if controls["uploaded_file"] is not None:
        if st.session_state.last_uploaded_name != controls["uploaded_file"].name:
            reset_dashboard_state()
            st.session_state.last_uploaded_name = controls["uploaded_file"].name

    video_placeholder, progress_placeholder, frame_placeholder = render_video_panel(
        output_col, controls["input_source"]
    )

    if controls["stop_clicked"]:
        try:
            requests.post(f"{API_URL}/reset", timeout=10)
        except Exception:
            pass
        add_alert("warning", "Stop requested. Processor state reset.")

    if controls["reset_clicked"]:
        try:
            requests.post(f"{API_URL}/reset", timeout=10)
        except Exception:
            pass
        reset_dashboard_state()
        add_alert("success", "Dashboard and processor reset")

    if controls["start_clicked"]:
        if controls["input_source"] == "Use Webcam":
            add_alert("warning", "Webcam live API is not exposed yet. Use Upload Video mode.")
        elif controls["uploaded_file"] is None:
            add_alert("warning", "Upload a video file before starting")
        else:
            start_processing(
                controls["uploaded_file"],
                controls["filters"],
                progress_placeholder,
                frame_placeholder,
            )
            if st.session_state.processing_result and st.session_state.processing_result.get(
                "output_file"
            ):
                video_placeholder.video(st.session_state.processing_result.get("output_file"))

    render_metrics(output_col)

    st.markdown("---")
    render_video_info()

    st.markdown("---")
    render_alerts()

    st.markdown("---")
    logs = fetch_latest_logs(limit=100)
    render_logs(logs)

    st.markdown("---")
    render_downloads()


if __name__ == "__main__":
    main()
