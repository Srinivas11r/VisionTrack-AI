"""
CSV export functionality with dynamic columns based on object type.
"""

import csv
import os
from datetime import datetime


def build_attributes_string(object_data):
    """Build clean attribute summary string by object class."""
    class_name = (object_data.get("class") or "").lower()
    attrs = object_data.get("attributes") or {}

    def valid(value):
        return value not in [None, "", "Unknown", "unknown"]

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

    if class_name in {"car", "truck", "bus", "motorcycle"}:
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


def export_tracks_to_csv(tracker, output_path="storage/outputs/tracking_results.csv"):
    """
    Export all tracked objects with their attributes to CSV.

    Args:
        tracker: ObjectTracker instance with tracked objects
        output_path: Path to save CSV file

    Returns:
        Path to saved CSV file
    """
    # Get all tracks with attributes
    all_tracks = tracker.get_all_tracks_with_attributes()

    if not all_tracks:
        print("⚠ No tracks to export")
        return None

    print(f"📊 Exporting {len(all_tracks)} tracks to CSV")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    column_order = ["Object_ID", "Type", "Attributes", "First_Seen", "Last_Seen", "Duration"]

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_order)
            writer.writeheader()

            # Write each track
            for track in all_tracks:
                object_data = {
                    "class": track.get("class", "Unknown"),
                    "attributes": track.get("attributes") or {},
                }
                row = {
                    "Object_ID": track["id"],
                    "Type": track.get("class", "Unknown"),
                    "Attributes": build_attributes_string(object_data),
                    "First_Seen": format_timestamp(track.get("first_seen")),
                    "Last_Seen": format_timestamp(track.get("last_seen")),
                    "Duration": round(track.get("duration", 0), 6),
                }

                writer.writerow(row)

        print(f"✓ CSV exported successfully: {output_path}")
        print(f"  Total tracks: {len(all_tracks)}")
        print(f"  Columns: {len(column_order)}")

        return output_path

    except Exception as e:
        print(f"✗ CSV export failed: {e}")
        return None


def format_timestamp(dt):
    """Format datetime object to readable string."""
    if dt is None:
        return "Unknown"

    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    return str(dt)


def export_summary_csv(tracker, output_path="storage/outputs/tracking_summary.csv"):
    """
    Export a summary CSV with aggregated statistics.

    Args:
        tracker: ObjectTracker instance
        output_path: Path to save summary CSV

    Returns:
        Path to saved CSV file
    """
    all_tracks = tracker.get_all_tracks_with_attributes()

    if not all_tracks:
        print("⚠ No tracks to summarize")
        return None

    # Aggregate statistics by object type
    summary = {}
    for track in all_tracks:
        obj_type = track.get("class", "Unknown")

        if obj_type not in summary:
            summary[obj_type] = {
                "Object_Type": obj_type,
                "Total_Count": 0,
                "Avg_Duration": [],
                "Attributes": [],
            }

        summary[obj_type]["Total_Count"] += 1
        summary[obj_type]["Avg_Duration"].append(track.get("duration", 0))

        # Collect attribute summaries
        if "attributes" in track:
            summary[obj_type]["Attributes"].append(
                build_attributes_string(
                    {"class": obj_type, "attributes": track.get("attributes") or {}}
                )
            )

    # Calculate averages and format
    summary_rows = []
    for obj_type, data in summary.items():
        avg_dur = (
            sum(data["Avg_Duration"]) / len(data["Avg_Duration"]) if data["Avg_Duration"] else 0
        )

        # Most common attribute summary
        from collections import Counter

        most_common_attr = (
            Counter(data["Attributes"]).most_common(1)[0][0] if data["Attributes"] else "N/A"
        )

        summary_rows.append(
            {
                "Object_Type": obj_type,
                "Total_Count": data["Total_Count"],
                "Avg_Duration": f"{avg_dur:.2f}s",
                "Most_Common_Attributes": most_common_attr,
            }
        )

    # Write summary CSV
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Object_Type", "Total_Count", "Avg_Duration", "Most_Common_Attributes"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"✓ Summary CSV exported: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Summary CSV export failed: {e}")
        return None
