import numpy as np

from backend.utils.filters import classify_size, detect_color, filter_detections


def test_detect_color_red_region():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:90, 10:90] = [0, 0, 255]
    assert detect_color(frame, [10, 10, 90, 90]) == "red"


def test_classify_size_dynamic():
    frame_area = 100 * 100
    assert classify_size([0, 0, 10, 10], frame_area=frame_area) == "small"
    assert classify_size([0, 0, 25, 25], frame_area=frame_area) == "medium"
    assert classify_size([0, 0, 95, 95], frame_area=frame_area) == "large"


def test_filter_detections_by_type_and_confidence():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        {"bbox": [10, 10, 40, 40], "confidence": 0.9, "class": "person"},
        {"bbox": [20, 20, 50, 50], "confidence": 0.2, "class": "person"},
        {"bbox": [30, 30, 60, 60], "confidence": 0.95, "class": "car"},
    ]
    specs = {"object_type": "person", "color": "all", "size": "all", "confidence": 0.5}
    filtered = filter_detections(detections, frame, specs)
    assert len(filtered) == 1
    assert filtered[0]["class"] == "person"
