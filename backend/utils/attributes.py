"""
Advanced attribute extraction for detected objects.
Extracts detailed attributes for persons and vehicles.
"""

from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_dominant_color(image_region, k=3):
    """
    Extract dominant color from image region using k-means clustering.

    Args:
        image_region: BGR image region
        k: Number of color clusters

    Returns:
        Human-readable color name
    """
    if image_region is None or image_region.size == 0:
        return "Unknown"

    try:
        # Reshape image to list of pixels
        pixels = image_region.reshape(-1, 3)

        # Remove very dark/black pixels (shadows, backgrounds)
        mask = np.all(pixels > 30, axis=1)
        if np.sum(mask) < 10:  # Not enough valid pixels
            return "Unknown"

        pixels = pixels[mask]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get the most common cluster center (dominant color)
        labels = kmeans.labels_
        dominant_cluster = Counter(labels).most_common(1)[0][0]
        dominant_bgr = kmeans.cluster_centers_[dominant_cluster]

        # Convert BGR to color name
        return bgr_to_color_name(dominant_bgr)

    except Exception as e:
        print(f"Error in color detection: {e}")
        return "Unknown"


def bgr_to_color_name(bgr):
    """
    Convert BGR values to human-readable color name.

    Args:
        bgr: BGR color values (0-255)

    Returns:
        Color name string
    """
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])

    # Convert to HSV for better color classification
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv[0], hsv[1], hsv[2]

    # Low saturation = grayscale colors
    if s < 30:
        if v < 50:
            return "Black"
        elif v < 130:
            return "Gray"
        else:
            return "White"

    # Classify by hue
    if h < 10 or h > 160:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 130:
        return "Blue"
    elif 130 <= h < 160:
        return "Purple"
    else:
        return "Unknown"


def extract_person_attributes(frame, bbox, yolo_model=None):
    """
    Extract detailed attributes for a detected person.

    Args:
        frame: Full video frame
        bbox: Bounding box [x1, y1, x2, y2]
        yolo_model: YOLO model for secondary detections (bags)

    Returns:
        Dictionary of person attributes
    """
    x1, y1, x2, y2 = map(int, bbox)
    person_crop = frame[y1:y2, x1:x2]

    if person_crop.size == 0:
        return {
            "Upper_Body_Color": "Unknown",
            "Lower_Body_Color": "Unknown",
            "Gender": "Unknown",
            "Age_Group": "Unknown",
            "Carrying_Bag": "Unknown",
        }

    height = y2 - y1
    width = x2 - x1

    # Split into upper and lower body (60/40 split)
    upper_split = int(height * 0.6)
    upper_body = person_crop[0:upper_split, :]
    lower_body = person_crop[upper_split:, :]

    # Extract colors
    upper_color = get_dominant_color(upper_body, k=3)
    lower_color = get_dominant_color(lower_body, k=3)

    # Gender estimation (simple heuristic based on aspect ratio and color)
    gender = estimate_gender(person_crop, height, width)

    # Age group estimation (based on height ratio and posture)
    age_group = estimate_age_group(height, width, frame.shape[0])

    # Bag detection using YOLO secondary check
    carrying_bag = "No"
    if yolo_model is not None:
        try:
            results = yolo_model(person_crop, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id].lower()
                    if any(
                        bag_type in class_name
                        for bag_type in ["backpack", "handbag", "suitcase", "bag"]
                    ):
                        carrying_bag = "Yes"
                        break
        except Exception as e:
            print(f"Bag detection error: {e}")
            carrying_bag = "Unknown"

    print(
        f"  → Person: Upper={upper_color}, Lower={lower_color}, Gender={gender}, Age={age_group}, Bag={carrying_bag}"
    )

    return {
        "Upper_Body_Color": upper_color,
        "Lower_Body_Color": lower_color,
        "Gender": gender,
        "Age_Group": age_group,
        "Carrying_Bag": carrying_bag,
    }


def estimate_gender(person_crop, height, width):
    """
    Simple gender estimation based on visual features.
    Note: This is a rough heuristic and may not be accurate.
    """
    try:
        aspect_ratio = height / width if width > 0 else 0

        # Check for long hair (top region has consistent color extending down)
        if height > 40:
            head_region = person_crop[0 : int(height * 0.25), :]
            if head_region.size > 0:
                # Simple heuristic: analyze aspect ratio and color distribution
                if aspect_ratio > 2.2:
                    return "Female"
                elif aspect_ratio < 1.8:
                    return "Male"

        return "Unknown"
    except Exception:
        return "Unknown"


def estimate_age_group(height, width, frame_height):
    """
    Estimate age group based on relative size in frame.
    """
    try:
        # Relative height in frame
        height_ratio = height / frame_height if frame_height > 0 else 0
        aspect_ratio = height / width if width > 0 else 0

        # Children are typically shorter with different proportions
        if height_ratio < 0.35 and aspect_ratio > 1.5:
            return "Child"
        # Seniors may have slightly hunched posture (lower aspect ratio)
        elif height_ratio > 0.5 and aspect_ratio < 1.6:
            return "Senior"
        else:
            return "Adult"
    except Exception:
        return "Unknown"


def extract_vehicle_attributes(frame, bbox, class_name, ocr_handler=None):
    """
    Extract detailed attributes for a detected vehicle.

    Args:
        frame: Full video frame
        bbox: Bounding box [x1, y1, x2, y2]
        class_name: YOLO detected class (car, truck, bus, etc.)
        ocr_handler: OCR handler for license plate detection

    Returns:
        Dictionary of vehicle attributes
    """
    x1, y1, x2, y2 = map(int, bbox)
    vehicle_crop = frame[y1:y2, x1:x2]

    if vehicle_crop.size == 0:
        return {
            "Vehicle_Color": "Unknown",
            "Vehicle_Type": class_name,
            "Vehicle_Company": "Unknown",
            "Body_Category": "Unknown",
            "License_Plate": "Unknown",
        }

    height = y2 - y1
    width = x2 - x1

    # Extract dominant vehicle color (upper half, avoiding ground/shadow)
    upper_vehicle = vehicle_crop[0 : int(height * 0.6), :]
    vehicle_color = get_dominant_color(upper_vehicle, k=3)

    # Classify body category based on aspect ratio and size
    body_category = classify_vehicle_body(height, width, class_name)
    vehicle_company = "Unknown"

    # License plate detection (lower-middle region)
    license_plate = "Unknown"
    if ocr_handler is not None:
        try:
            # Extract plate region (bottom 30%, middle 60% horizontally)
            plate_y_start = int(height * 0.7)
            plate_x_start = int(width * 0.2)
            plate_x_end = int(width * 0.8)
            plate_region = vehicle_crop[plate_y_start:, plate_x_start:plate_x_end]

            if plate_region.size > 0:
                license_plate = ocr_handler.detect_plate(plate_region)
        except Exception as e:
            print(f"License plate OCR error: {e}")
            license_plate = "Unknown"

    print(
        f"  → Vehicle: Color={vehicle_color}, Type={class_name}, Body={body_category}, Plate={license_plate}"
    )

    return {
        "Vehicle_Color": vehicle_color,
        "Vehicle_Type": class_name,
        "Vehicle_Company": vehicle_company,
        "Body_Category": body_category,
        "License_Plate": license_plate,
    }


def extract_bag_attributes(frame, bbox, class_name):
    """Extract attributes for bag-like objects."""
    x1, y1, x2, y2 = map(int, bbox)
    obj_crop = frame[y1:y2, x1:x2]

    if obj_crop.size == 0:
        return {"Bag_Color": "Unknown", "Bag_Type": class_name}

    return {"Bag_Color": get_dominant_color(obj_crop, k=3), "Bag_Type": class_name}


def extract_animal_attributes(frame, bbox, class_name):
    """Extract attributes for animal objects."""
    x1, y1, x2, y2 = map(int, bbox)
    obj_crop = frame[y1:y2, x1:x2]

    if obj_crop.size == 0:
        return {"Animal_Species": class_name, "Object_Color": "Unknown"}

    return {"Animal_Species": class_name, "Object_Color": get_dominant_color(obj_crop, k=3)}


def extract_generic_attributes(frame, bbox, class_name):
    """Fallback attribute extraction for any other detected class."""
    x1, y1, x2, y2 = map(int, bbox)
    obj_crop = frame[y1:y2, x1:x2]

    if obj_crop.size == 0:
        return {"Object_Color": "Unknown", "Object_Subtype": class_name}

    return {"Object_Color": get_dominant_color(obj_crop, k=3), "Object_Subtype": class_name}


def classify_vehicle_body(height, width, class_name):
    """
    Classify vehicle body type based on dimensions.
    """
    try:
        aspect_ratio = width / height if height > 0 else 0

        if class_name in ["truck", "bus"]:
            return "Large_Commercial"
        elif class_name == "motorcycle":
            return "Two_Wheeler"
        elif class_name == "car":
            # Aspect ratio helps distinguish car types
            if aspect_ratio > 2.2:
                return "SUV"
            elif aspect_ratio > 1.8:
                return "Sedan"
            else:
                return "Small_Car"
        else:
            return class_name.title()
    except Exception:
        return "Unknown"
