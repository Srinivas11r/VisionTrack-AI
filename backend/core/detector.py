import logging

import cv2
from ultralytics import YOLO

logger = logging.getLogger("app.detector")


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.5):
        logger.info("Loading model: %s", model_name)
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        logger.info("Model loaded")

    def detect(self, frame):
        """
        Run detection on a single frame
        Returns list of detections with bbox, class, confidence
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Handle both CPU and GPU tensors
                bbox_tensor = box.xyxy[0]
                if hasattr(bbox_tensor, "cpu"):
                    x1, y1, x2, y2 = bbox_tensor.cpu().numpy()
                else:
                    x1, y1, x2, y2 = bbox_tensor.numpy()

                conf_tensor = box.conf[0]
                if hasattr(conf_tensor, "cpu"):
                    conf = float(conf_tensor.cpu().numpy())
                else:
                    conf = float(conf_tensor.numpy())

                cls_tensor = box.cls[0]
                if hasattr(cls_tensor, "cpu"):
                    cls = int(cls_tensor.cpu().numpy())
                else:
                    cls = int(cls_tensor.numpy())

                class_name = self.model.names[cls]

                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "class": class_name,
                        "class_id": cls,
                    }
                )

        return detections

    def detect_video(self, video_path, output_path=None):
        """
        Process entire video file
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error("Error opening video: %s", video_path)
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            detections = self.detect(frame)

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class']} {det['confidence']:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            if writer:
                writer.write(frame)

            if frame_count % 30 == 0:
                logger.info("Processed %s frames", frame_count)

        cap.release()
        if writer:
            writer.release()

        logger.info("Video processing complete. Total frames: %s", frame_count)
        return frame_count


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    detector = ObjectDetector()
    logger.info("Detector initialized successfully")
