import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class TrackLog(Base):
    __tablename__ = "track_logs"

    id = Column(Integer, primary_key=True)
    track_id = Column(Integer, nullable=False)
    object_class = Column(String, nullable=False)
    color = Column(String)
    size = Column(String)
    gender = Column(String)
    age_group = Column(String)
    shirt_color = Column(String)
    pant_color = Column(String)
    number_plate = Column(String)
    vehicle_color = Column(String)
    vehicle_company = Column(String)
    body_type = Column(String)
    bag_color = Column(String)
    bag_type = Column(String)
    animal_species = Column(String)
    object_color = Column(String)
    attributes_json = Column(Text)
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    duration = Column(Float)
    frame_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class Database:
    def __init__(self, db_url="sqlite:///tracking.db"):
        if db_url.startswith("sqlite:///"):
            sqlite_path = db_url.replace("sqlite:///", "", 1)
            path_obj = Path(sqlite_path).resolve()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{str(path_obj)}"

        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self._ensure_dynamic_columns()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def _ensure_dynamic_columns(self):
        """Apply lightweight in-place migration for dynamic attribute fields."""
        if self.engine.dialect.name != "sqlite":
            return

        required = {
            "gender": "TEXT",
            "age_group": "TEXT",
            "shirt_color": "TEXT",
            "pant_color": "TEXT",
            "number_plate": "TEXT",
            "vehicle_color": "TEXT",
            "vehicle_company": "TEXT",
            "body_type": "TEXT",
            "bag_color": "TEXT",
            "bag_type": "TEXT",
            "animal_species": "TEXT",
            "object_color": "TEXT",
            "attributes_json": "TEXT",
        }

        with self.engine.connect() as conn:
            rows = conn.exec_driver_sql("PRAGMA table_info(track_logs)").fetchall()
            existing = {row[1] for row in rows}
            for col, col_type in required.items():
                if col not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE track_logs ADD COLUMN {col} {col_type}")

    def _normalized_attributes(self, log_data):
        attrs = log_data.get("attributes") or {}
        return {
            "gender": log_data.get("gender") or attrs.get("gender", ""),
            "age_group": log_data.get("age_group") or attrs.get("age_group", ""),
            "shirt_color": log_data.get("shirt_color") or attrs.get("shirt_color", ""),
            "pant_color": log_data.get("pant_color") or attrs.get("pant_color", ""),
            "number_plate": log_data.get("number_plate") or attrs.get("number_plate", ""),
            "vehicle_color": log_data.get("vehicle_color") or attrs.get("vehicle_color", ""),
            "vehicle_company": log_data.get("vehicle_company") or attrs.get("vehicle_company", ""),
            "body_type": log_data.get("body_type") or attrs.get("body_type", ""),
            "bag_color": log_data.get("bag_color") or attrs.get("bag_color", ""),
            "bag_type": log_data.get("bag_type") or attrs.get("bag_type", ""),
            "animal_species": log_data.get("animal_species") or attrs.get("animal_species", ""),
            "object_color": log_data.get("object_color") or attrs.get("object_color", ""),
            "attributes_json": json.dumps(attrs, ensure_ascii=False),
        }

    def build_attributes_string(self, object_data):
        """Build minimal attributes string for storage/display."""
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
                attrs.get("vehicle_color")
                or attrs.get("Vehicle_Color")
                or attrs.get("object_color")
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

    def add_track_log(self, log_data):
        """Add a track log entry"""
        attrs = self._normalized_attributes(log_data)
        track_id = log_data.get("Object_ID", log_data.get("id"))
        object_class = log_data.get("Type", log_data.get("class", "unknown"))
        first_seen_value = log_data.get("First_Seen", log_data.get("first_seen"))
        last_seen_value = log_data.get("Last_Seen", log_data.get("last_seen"))
        duration_value = log_data.get("Duration", log_data.get("duration", 0))
        if isinstance(duration_value, str):
            try:
                duration_value = float(duration_value)
            except Exception:
                duration_value = 0

        object_data = {"class": object_class, "attributes": log_data.get("attributes") or {}}

        log = TrackLog(
            track_id=track_id,
            object_class=object_class,
            color=log_data.get("color")
            or attrs.get("object_color")
            or log_data.get("vehicle_color")
            or log_data.get("shirt_color"),
            size=log_data.get("size"),
            gender=attrs["gender"],
            age_group=attrs["age_group"],
            shirt_color=attrs["shirt_color"],
            pant_color=attrs["pant_color"],
            number_plate=attrs["number_plate"],
            vehicle_color=attrs["vehicle_color"],
            vehicle_company=attrs["vehicle_company"],
            body_type=attrs["body_type"],
            bag_color=attrs["bag_color"],
            bag_type=attrs["bag_type"],
            animal_species=attrs["animal_species"],
            object_color=self.build_attributes_string(object_data),
            attributes_json=attrs["attributes_json"],
            first_seen=datetime.fromisoformat(first_seen_value),
            last_seen=datetime.fromisoformat(last_seen_value),
            duration=duration_value,
            frame_count=log_data.get("frame_count", 0),
        )
        self.session.add(log)
        self.session.commit()
        return log.id

    def get_all_logs(self, limit=100):
        """Get all track logs"""
        logs = self.session.query(TrackLog).order_by(TrackLog.created_at.desc()).limit(limit).all()
        return [
            {
                "Object_ID": log.track_id,
                "Type": log.object_class,
                "Attributes": log.object_color or "Unknown",
                "First_Seen": log.first_seen.isoformat(),
                "Last_Seen": log.last_seen.isoformat(),
                "Duration": round(log.duration or 0, 6),
            }
            for log in logs
        ]

    def clear_logs(self):
        """Clear all logs"""
        self.session.query(TrackLog).delete()
        self.session.commit()


if __name__ == "__main__":
    # Test database
    db = Database("test_tracking.db")
    print("Database initialized")

    # Test adding log
    test_log = {
        "id": 1,
        "class": "person",
        "color": "blue",
        "size": "medium",
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
        "duration": 5.0,
        "frame_count": 150,
    }
    db.add_track_log(test_log)
    print("Test log added")

    logs = db.get_all_logs()
    print(f"Retrieved {len(logs)} logs")
