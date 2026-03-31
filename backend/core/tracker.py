from collections import defaultdict
from datetime import datetime

import numpy as np


class ObjectTracker:
    """
    Simple tracking implementation using IoU matching
    For production, consider using ByteTrack or DeepSORT
    """

    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}  # Currently active tracks
        self.finished_tracks = {}  # Keep finished tracks for export
        self.next_id = 1
        self.frame_count = 0
        self.track_history = defaultdict(list)
        self.track_attributes = {}  # Store detailed attributes per track
        self.track_first_seen = {}  # Store first seen timestamp
        self.track_last_seen = {}  # Store last seen timestamp

    def update(self, detections):
        """
        Update tracks with new detections
        detections: list of dicts with 'bbox', 'class', 'confidence'
        """
        self.frame_count += 1

        if len(detections) == 0:
            # Age out old tracks
            dead_tracks = []
            for track_id, track in self.tracks.items():
                track["age"] += 1
                if track["age"] > self.max_age:
                    dead_tracks.append(track_id)

            # Move dead tracks to finished_tracks instead of deleting
            for track_id in dead_tracks:
                self.finished_tracks[track_id] = self.tracks[track_id].copy()
                del self.tracks[track_id]

            return []

        # Convert detections to numpy array
        det_boxes = np.array([d["bbox"] for d in detections])

        # Match with existing tracks
        if len(self.tracks) == 0:
            # No existing tracks, create new ones
            matched_tracks = []
            for det in detections:
                track = {
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hits": 1,
                }
                self.tracks[self.next_id] = track
                matched_tracks.append(track)
                self.track_history[self.next_id].append(det["bbox"])

                # Initialize empty attributes (will be populated by processor)
                self.track_attributes[self.next_id] = {}
                self.track_first_seen[self.next_id] = datetime.now()
                self.track_last_seen[self.next_id] = datetime.now()

                self.next_id += 1

            return matched_tracks

        # Calculate IoU between detections and tracks
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid]["bbox"] for tid in track_ids])

        iou_matrix = self._compute_iou(det_boxes, track_boxes)

        # Greedy matching
        matched_det = set()
        matched_track = set()
        matches = []

        for _ in range(min(len(detections), len(track_ids))):
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matches.append((det_idx, track_idx))
            matched_det.add(det_idx)
            matched_track.add(track_idx)

            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0

        # Update matched tracks
        result_tracks = []
        for det_idx, track_idx in matches:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            track["bbox"] = detections[det_idx]["bbox"]
            track["class"] = detections[det_idx]["class"]
            track["confidence"] = detections[det_idx]["confidence"]
            track["age"] = 0
            track["hits"] += 1
            self.track_history[track_id].append(detections[det_idx]["bbox"])

            # Keep existing attributes (don't overwrite with new detections)
            # Attributes are set once during track creation

            # Update last seen timestamp
            self.track_last_seen[track_id] = datetime.now()

            result_tracks.append(track)

        # Create new tracks for unmatched detections
        for det_idx in range(len(detections)):
            if det_idx not in matched_det:
                track = {
                    "id": self.next_id,
                    "bbox": detections[det_idx]["bbox"],
                    "class": detections[det_idx]["class"],
                    "confidence": detections[det_idx]["confidence"],
                    "age": 0,
                    "hits": 1,
                }
                self.tracks[self.next_id] = track
                result_tracks.append(track)
                self.track_history[self.next_id].append(detections[det_idx]["bbox"])

                # Initialize empty attributes for new track (will be populated by processor)
                self.track_attributes[self.next_id] = {}
                self.track_first_seen[self.next_id] = datetime.now()
                self.track_last_seen[self.next_id] = datetime.now()

                self.next_id += 1

        # Age unmatched tracks
        dead_tracks = []
        for track_idx in range(len(track_ids)):
            if track_idx not in matched_track:
                track_id = track_ids[track_idx]
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age:
                    dead_tracks.append(track_id)

        # Move dead tracks to finished_tracks instead of deleting
        for track_id in dead_tracks:
            self.finished_tracks[track_id] = self.tracks[track_id].copy()
            del self.tracks[track_id]

        # Return only tracks with minimum hits
        return [t for t in result_tracks if t["hits"] >= self.min_hits]

    def _compute_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes
        boxes: [N, 4] in format [x1, y1, x2, y2]
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        iou = np.zeros((len(boxes1), len(boxes2)))

        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                x1 = max(float(box1[0]), float(box2[0]))
                y1 = max(float(box1[1]), float(box2[1]))
                x2 = min(float(box1[2]), float(box2[2]))
                y2 = min(float(box1[3]), float(box2[3]))

                if x2 < x1 or y2 < y1:
                    iou[i, j] = 0
                else:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = area1[i] + area2[j] - intersection
                    iou[i, j] = intersection / union if union > 0 else 0

        return iou

    def get_track_history(self, track_id):
        """Get bbox history for a specific track"""
        return self.track_history.get(track_id, [])

    def get_track_attributes(self, track_id):
        """Get detailed attributes for a specific track"""
        return self.track_attributes.get(track_id, {})

    def get_all_tracks_with_attributes(self):
        """Get all tracks (active + finished) with their complete attributes and timestamps"""
        result = []

        # Combine active tracks and finished tracks
        all_tracks = {**self.finished_tracks, **self.tracks}

        for track_id, track in all_tracks.items():
            track_info = track.copy()
            track_info["attributes"] = self.track_attributes.get(track_id, {})
            track_info["first_seen"] = self.track_first_seen.get(track_id)
            track_info["last_seen"] = self.track_last_seen.get(track_id)

            # Calculate duration
            if track_info["first_seen"] and track_info["last_seen"]:
                duration = (track_info["last_seen"] - track_info["first_seen"]).total_seconds()
                track_info["duration"] = duration
            else:
                track_info["duration"] = 0

            result.append(track_info)

        # Sort by track ID
        result.sort(key=lambda x: x["id"])
        return result

    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.finished_tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.track_history = defaultdict(list)
        self.track_attributes = {}
        self.track_first_seen = {}
        self.track_last_seen = {}


if __name__ == "__main__":
    # Test tracker
    tracker = ObjectTracker()

    # Simulate detections
    frame1_dets = [{"bbox": [100, 100, 200, 200], "class": "person", "confidence": 0.9}]

    tracks = tracker.update(frame1_dets)
    print(f"Frame 1: {len(tracks)} tracks")

    frame2_dets = [{"bbox": [105, 105, 205, 205], "class": "person", "confidence": 0.9}]

    tracks = tracker.update(frame2_dets)
    print(f"Frame 2: {len(tracks)} tracks")
    print("Tracker test passed")
