import logging
import time
from datetime import datetime

import cv2
from utils.attributes import (
    extract_animal_attributes,
    extract_bag_attributes,
    extract_generic_attributes,
    extract_person_attributes,
    extract_vehicle_attributes,
)
from utils.filters import filter_detections
from utils.ocr_handler import OCRHandler

from .detector import ObjectDetector
from .tracker import ObjectTracker

logger = logging.getLogger("app.processor")


def build_attributes_string(object_data):
    """Build a clean, minimal attributes string based on object type."""
    class_name = (object_data.get("class") or "").lower()
    attrs = object_data.get("attributes") or {}

    def valid(value):
        return value not in [None, "", "Unknown", "unknown"]

    vehicle_classes = {"car", "truck", "bus", "motorcycle"}

    if class_name == "person":
        upper = attrs.get("shirt_color") or attrs.get("Upper_Body_Color")
        lower = attrs.get("pant_color") or attrs.get("Lower_Body_Color")
        gender = attrs.get("gender") or attrs.get("Gender")
        parts = []
        if valid(upper):
            parts.append(f"Upper: {upper}")
        if valid(lower):
            parts.append(f"Lower: {lower}")
        if valid(gender):
            parts.append(f"Gender: {gender}")
        return ", ".join(parts) if parts else "Unknown"

    if class_name in vehicle_classes:
        color = (
            attrs.get("vehicle_color") or attrs.get("Vehicle_Color") or attrs.get("object_color")
        )
        plate = attrs.get("number_plate") or attrs.get("License_Plate")
        category = attrs.get("body_type") or attrs.get("Body_Category")
        parts = []
        if valid(color):
            parts.append(str(color))
        if valid(plate):
            parts.append(f"Plate: {plate}")
        if valid(category):
            parts.append(f"Category: {category}")
        return ", ".join(parts) if parts else "Unknown"

    color = attrs.get("object_color") or attrs.get("color")
    if valid(color):
        return f"Color: {color}"

    return "Unknown"


class VideoProcessor:
    """
    Main processor that combines detection, tracking, and filtering
    """

    BAG_CLASSES = {"bag", "backpack", "handbag", "suitcase"}
    ANIMAL_CLASSES = {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }

    def __init__(
        self,
        model_name="yolov8n.pt",
        performance_mode=False,
        ocr_enabled=True,
        max_objects_per_frame=50,
    ):
        logger.info("Initializing Video Processor")
        self.detector = ObjectDetector(model_name=model_name)
        self.tracker = ObjectTracker()
        self.specifications = {
            "object_type": "all",
            "color": "all",
            "size": "all",
            "confidence": 0.25,  # Lowered default for better detection
        }
        self.track_logs = {}
        self.debug_warnings = []
        self.performance_mode = performance_mode  # Skip OCR, frame-skip for speed
        self.frame_skip = (
            3 if performance_mode else 1
        )  # Process every 3rd frame in performance mode
        self.max_objects_per_frame = max_objects_per_frame  # Limit objects to prevent slowdown

        # Initialize OCR handler for license plate detection (disabled in performance mode)
        if not performance_mode and ocr_enabled:
            try:
                self.ocr_handler = OCRHandler(use_easyocr=True)
            except Exception as e:
                logger.warning("OCR initialization skipped: %s", e)
                self.ocr_handler = None
        else:
            logger.info(
                "OCR disabled (performance_mode=%s, ocr_enabled=%s, frame skip=%s)",
                performance_mode,
                ocr_enabled,
                self.frame_skip,
            )
            self.ocr_handler = None

        logger.info("Video Processor initialized (max %s objects/frame)", max_objects_per_frame)

    def set_specifications(self, specs):
        """Update filtering specifications"""
        self.specifications.update(specs)
        logger.info("Specifications updated: %s", self.specifications)

        # PART 3: Confidence warning
        if self.specifications.get("confidence", 0.25) > 0.7:
            warning = "⚠ High confidence (>0.7) may remove valid detections"
            logger.warning(warning)
            self.debug_warnings.append(warning)

    def _normalize_attributes(self, class_name, attributes):
        """Normalize extracted attributes into a stable dynamic schema."""
        normalized = {}

        if class_name == "person":
            normalized["gender"] = attributes.get("Gender", "Unknown")
            normalized["age_group"] = attributes.get("Age_Group", "Unknown")
            normalized["shirt_color"] = attributes.get("Upper_Body_Color", "Unknown")
            normalized["pant_color"] = attributes.get("Lower_Body_Color", "Unknown")
            normalized["object_color"] = normalized["shirt_color"]
        elif class_name in ["car", "truck", "bus", "motorcycle"]:
            normalized["number_plate"] = attributes.get("License_Plate", "Unknown")
            normalized["vehicle_color"] = attributes.get("Vehicle_Color", "Unknown")
            normalized["vehicle_company"] = attributes.get("Vehicle_Company", "Unknown")
            normalized["body_type"] = attributes.get("Body_Category", "Unknown")
            normalized["object_color"] = normalized["vehicle_color"]
        elif class_name in self.BAG_CLASSES:
            normalized["bag_color"] = attributes.get("Bag_Color", "Unknown")
            normalized["bag_type"] = attributes.get("Bag_Type", class_name)
            normalized["object_color"] = normalized["bag_color"]
        elif class_name in self.ANIMAL_CLASSES:
            normalized["animal_species"] = attributes.get("Animal_Species", class_name)
            normalized["color"] = attributes.get("Object_Color", "Unknown")
            normalized["object_color"] = normalized["color"]
        else:
            normalized["object_color"] = attributes.get("Object_Color", "Unknown")

        normalized.update({"class": class_name, "raw_attributes": attributes})
        return normalized

    def _extract_dynamic_attributes(self, frame, bbox, class_name):
        """Extract attributes based on class type (once per new track)."""
        if class_name == "person":
            raw_attrs = extract_person_attributes(frame, bbox, yolo_model=self.detector.model)
        elif class_name in ["car", "truck", "bus", "motorcycle"]:
            raw_attrs = extract_vehicle_attributes(
                frame, bbox, class_name, ocr_handler=self.ocr_handler
            )
        elif class_name in self.BAG_CLASSES:
            raw_attrs = extract_bag_attributes(frame, bbox, class_name)
        elif class_name in self.ANIMAL_CLASSES:
            raw_attrs = extract_animal_attributes(frame, bbox, class_name)
        else:
            raw_attrs = extract_generic_attributes(frame, bbox, class_name)

        return self._normalize_attributes(class_name, raw_attrs)

    def process_frame(self, frame, frame_number=0):
        """
        Process single frame: detect -> filter -> track -> extract attributes
        Returns: annotated frame, active tracks
        """
        # PART 2: Detection debugging
        detections = self.detector.detect(frame)
        logger.debug("Frame %s: %s raw detections found", frame_number, len(detections))

        # Apply filters
        filtered = filter_detections(detections, frame, self.specifications)
        logger.debug("Frame %s: %s detections after filtering", frame_number, len(filtered))

        # Limit objects per frame to prevent slowdown with crowded scenes
        if len(filtered) > self.max_objects_per_frame:
            warning = f"⚠ Frame {frame_number}: Too many objects ({len(filtered)}), limiting to {self.max_objects_per_frame}"
            logger.warning(warning)
            self.debug_warnings.append(warning)
            # Keep highest confidence detections
            filtered = sorted(filtered, key=lambda x: x["confidence"], reverse=True)[
                : self.max_objects_per_frame
            ]

        # PART 7: Tracker input validation
        if len(filtered) == 0 and len(detections) > 0:
            warning = f"⚠ Frame {frame_number}: Filters rejected all {len(detections)} detections"
            logger.warning(warning)
            self.debug_warnings.append(warning)

        # Update tracker first to get track assignments
        tracks = self.tracker.update(filtered)
        logger.debug("Frame %s: %s active tracks", frame_number, len(tracks))

        # Skip attribute extraction in performance mode or on frame-skip cycles
        skip_attributes = self.performance_mode and (frame_number % self.frame_skip != 0)

        # Extract attributes ONLY for NEW tracks (tracks without attributes yet)
        if not skip_attributes:
            for track in tracks:
                track_id = track["id"]

                # Skip if this track already has attributes
                if (
                    track_id in self.tracker.track_attributes
                    and self.tracker.track_attributes[track_id]
                ):
                    continue

                class_name = track["class"].lower()
                bbox = track["bbox"]

                logger.debug("[NEW TRACK %s] Extracting attributes for %s", track_id, class_name)
                attributes = self._extract_dynamic_attributes(frame, bbox, class_name)

                # Store attributes in tracker
                self.tracker.track_attributes[track_id] = attributes
                logger.debug(
                    "Stored attributes for track %s: %s", track_id, list(attributes.keys())
                )

        # Log tracks
        for track in tracks:
            track_id = track["id"]
            if track_id not in self.track_logs:
                self.track_logs[track_id] = {
                    "id": track_id,
                    "class": track["class"],
                    "color": track.get("color"),
                    "size": track.get("size"),
                    "first_seen": datetime.now(),
                    "last_seen": datetime.now(),
                    "frame_count": 1,
                }
            else:
                self.track_logs[track_id]["last_seen"] = datetime.now()
                self.track_logs[track_id]["frame_count"] += 1

        # Draw on frame
        annotated = self._draw_tracks(frame.copy(), tracks)

        return annotated, tracks

    def _draw_tracks(self, frame, tracks):
        """Draw bounding boxes and labels for tracks"""
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            # Ensure bbox coordinates are integers for cv2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = track["id"]
            label = f"ID:{track_id} | {track['class']} | {track['confidence']:.2f}"

            # Add color and size info if available
            if track.get("color"):
                label += f" | {track['color']}"
            if track.get("size"):
                label += f" | {track['size']}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1
            )

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def process_video(self, video_path, output_path=None, skip_frames=0, progress_callback=None):
        """
        Process entire video file with comprehensive debugging
        """
        logger.info("Processing video: %s", video_path)

        if progress_callback:
            progress_callback(
                {
                    "progress": 0,
                    "current_frame": 0,
                    "total_frames": 0,
                    "message": "Initializing video processing",
                }
            )

        # PART 1: Video Loading Fix
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            error_msg = f"❌ ERROR: Unable to open video file: {video_path}"
            logger.error(error_msg)
            return {"error": error_msg, "logs": [], "metadata": None, "warnings": []}

        # Extract video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration": duration,
            "frame_area": width * height,
        }

        logger.info(
            "Video opened: resolution=%sx%s fps=%.2f total_frames=%s duration=%.2fs frame_area=%s",
            width,
            height,
            fps,
            total_frames,
            duration,
            metadata["frame_area"],
        )

        if progress_callback:
            progress_callback(
                {
                    "progress": 0,
                    "current_frame": 0,
                    "total_frames": total_frames,
                    "message": "Video loaded. Starting detection...",
                }
            )

        # Initialize video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))
            if writer.isOpened():
                logger.info("Output video writer initialized: %s", output_path)
            else:
                logger.warning("Could not initialize video writer")

        frame_count = 0
        processed_count = 0
        start_time = time.time()
        zero_detection_frames = 0

        # Reset warnings for this session
        self.debug_warnings = []

        logger.info("Starting frame processing")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if needed
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue

            processed_count += 1
            annotated, tracks = self.process_frame(frame, frame_number=frame_count)

            if len(tracks) == 0:
                zero_detection_frames += 1

            if writer and writer.isOpened():
                writer.write(annotated)

            # Progress update every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: frame %s/%s (%.1f%%) fps=%.1f active_tracks=%s unique_ids=%s",
                    frame_count,
                    total_frames,
                    (frame_count / total_frames * 100) if total_frames else 0,
                    fps_actual,
                    len(tracks),
                    len(self.track_logs),
                )

            if (
                progress_callback
                and total_frames > 0
                and (frame_count % 10 == 0 or frame_count == total_frames)
            ):
                progress_callback(
                    {
                        "progress": min(99, int((frame_count / total_frames) * 100)),
                        "current_frame": frame_count,
                        "total_frames": total_frames,
                        "message": f"Processing frame {frame_count}/{total_frames}",
                    }
                )

        cap.release()
        if writer:
            writer.release()

        elapsed = time.time() - start_time

        logger.info(
            "Processing complete: frames_read=%s frames_processed=%s processing_time=%.2fs avg_fps=%.2f unique_objects=%s zero_detection_frames=%s/%s",
            frame_count,
            processed_count,
            elapsed,
            (frame_count / elapsed) if elapsed > 0 else 0,
            len(self.track_logs),
            zero_detection_frames,
            processed_count,
        )

        # PART 2: Warning if no detections
        if zero_detection_frames > processed_count * 0.8:  # 80% of frames had no detections
            warning = (
                f"⚠ WARNING: {zero_detection_frames}/{processed_count} frames had no detections. "
                f"Try lowering confidence threshold (current: {self.specifications.get('confidence', 0.25)})"
            )
            logger.warning(warning)
            self.debug_warnings.append(warning)

        # PART 7: Filter strictness warning
        if len(self.track_logs) == 0:
            warning = (
                "⚠ WARNING: No objects tracked! Filters may be too strict. "
                "Try: Size='all', Color='all', Object Type='all'"
            )
            logger.warning(warning)
            self.debug_warnings.append(warning)

        if self.debug_warnings:
            logger.warning("Debug warnings encountered:")
            for w in self.debug_warnings:
                logger.warning("  %s", w)

        if progress_callback:
            progress_callback(
                {
                    "progress": 100,
                    "current_frame": frame_count,
                    "total_frames": total_frames,
                    "message": "Processing complete",
                }
            )

        return {
            "logs": self.get_logs(),
            "metadata": metadata,
            "warnings": self.debug_warnings,
            "stats": {
                "frames_read": frame_count,
                "frames_processed": processed_count,
                "processing_time": elapsed,
                "avg_fps": frame_count / elapsed if elapsed > 0 else 0,
                "unique_objects": len(self.track_logs),
                "zero_detection_frames": zero_detection_frames,
            },
        }

    def process_webcam(self, output_path=None):
        """
        Process live webcam feed with debugging
        Press 'q' to quit
        """
        logger.info("Starting webcam feed")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            error_msg = "❌ ERROR: Unable to open webcam"
            logger.error(error_msg)
            return {"error": error_msg, "logs": [], "warnings": []}

        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info("Webcam opened: %sx%s", width, height)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                logger.info("Recording to: %s", output_path)

        frame_count = 0
        self.debug_warnings = []

        logger.info("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated, tracks = self.process_frame(frame, frame_number=frame_count)

            # Display stats on frame
            stats_text = (
                f"Frame: {frame_count} | Tracks: {len(tracks)} | IDs: {len(self.track_logs)}"
            )
            cv2.putText(
                annotated, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow("Object Tracking", annotated)

            if writer and writer.isOpened():
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        logger.info("Webcam closed: %s frames processed", frame_count)
        logger.info("Unique objects tracked: %s", len(self.track_logs))

        return {
            "logs": self.get_logs(),
            "warnings": self.debug_warnings,
            "stats": {"frames_processed": frame_count, "unique_objects": len(self.track_logs)},
        }

    def get_logs(self):
        """Return tracking logs"""
        logs = []
        for track_id, log in self.track_logs.items():
            duration = (log["last_seen"] - log["first_seen"]).total_seconds()
            attrs = self.tracker.track_attributes.get(track_id, {})
            object_data = {"class": log["class"], "attributes": attrs}
            logs.append(
                {
                    "Object_ID": log["id"],
                    "Type": log["class"],
                    "Attributes": build_attributes_string(object_data),
                    "First_Seen": log["first_seen"].isoformat(),
                    "Last_Seen": log["last_seen"].isoformat(),
                    "Duration": round(duration, 6),
                    "frame_count": log["frame_count"],
                    "attributes": attrs,
                }
            )
        return logs

    def get_warnings(self):
        """Return debug warnings"""
        return self.debug_warnings

    def reset(self):
        """Reset processor state"""
        self.tracker.reset()
        self.track_logs = {}
        self.debug_warnings = []
        logger.info("Processor reset")
