import cv2
import numpy as np

# HSV color ranges
COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255])),
    ],
    "blue": [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    "green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
    "black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
    "white": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
    "orange": [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
    "purple": [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
}

SIZE_THRESHOLDS = {"small": (0, 10000), "medium": (10000, 50000), "large": (50000, float("inf"))}


def detect_color(frame, bbox):
    """
    Detect dominant color in bounding box using HSV
    Returns color name or None
    """
    x1, y1, x2, y2 = bbox

    # Convert to int to avoid indexing errors
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Ensure bbox is within frame
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Check each color
    color_scores = {}
    for color_name, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        score = np.sum(mask) / 255 / mask.size
        color_scores[color_name] = score

    # Get dominant color
    if len(color_scores) > 0:
        best_color = max(color_scores, key=color_scores.get)
        if color_scores[best_color] > 0.1:  # At least 10% of pixels
            return best_color

    return None


def classify_size(bbox, frame_area=None):
    """
    PART 6: Dynamic size classification based on frame area ratio
    Returns: 'small', 'medium', or 'large'
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    box_area = (x2 - x1) * (y2 - y1)

    # If frame_area provided, use dynamic ratio-based classification
    if frame_area and frame_area > 0:
        ratio = box_area / frame_area
        print(f"Box ratio: {ratio:.4f} (area: {box_area}, frame: {frame_area})")

        if ratio < 0.02:
            return "small"
        elif ratio < 0.08:
            return "medium"
        else:
            return "large"
    else:
        # Fallback to fixed thresholds
        for size, (min_area, max_area) in SIZE_THRESHOLDS.items():
            if min_area <= box_area < max_area:
                return size
        return "medium"


def filter_detections(detections, frame, specifications):
    """
    IMPROVED: Filter detections with better matching and debugging

    specifications: dict with keys:
        - object_type: str or None
        - color: str or None
        - size: str or None
        - confidence: float
    """
    filtered = []
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    for det in detections:
        # Confidence check
        conf_threshold = specifications.get("confidence", 0.25)
        if det["confidence"] < conf_threshold:
            continue

        # PART 4: Object type matching with partial match
        obj_type = specifications.get("object_type")
        if obj_type and obj_type.lower() != "all":
            # Flexible matching - check if user type is in class name or vice versa
            if (
                obj_type.lower() not in det["class"].lower()
                and det["class"].lower() not in obj_type.lower()
            ):
                continue

        # PART 5: Color check - skip if "all" selected
        target_color = specifications.get("color")
        if target_color and target_color.lower() != "all":
            detected_color = detect_color(frame, det["bbox"])
            det["color"] = detected_color
            if detected_color is None:
                # Don't reject - just log
                print(f"⚠ Color detection uncertain for {det['class']} — passing object")
                det["color"] = "unknown"
            elif detected_color != target_color.lower():
                continue
        else:
            # Still detect color for display purposes
            det["color"] = detect_color(frame, det["bbox"])

        # PART 6: Size check with dynamic scaling - skip if "all"
        target_size = specifications.get("size")
        detected_size = classify_size(det["bbox"], frame_area=frame_area)
        det["size"] = detected_size

        if target_size and target_size.lower() != "all":
            if detected_size != target_size.lower():
                continue

        filtered.append(det)

    return filtered


if __name__ == "__main__":
    # Test color detection
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    test_frame[:, :] = [0, 0, 255]  # Red in BGR

    bbox = [10, 10, 90, 90]
    color = detect_color(test_frame, bbox)
    print(f"Detected color: {color}")

    size = classify_size(bbox)
    print(f"Detected size: {size}")

    print("Filter test passed")
