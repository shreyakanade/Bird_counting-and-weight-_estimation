from typing import List, Dict, Any


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0


class Tracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 5):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]] | None):
        if not detections:
            detections = []

        updated_tracks = {}
        assigned = set()

        for track_id, track in self.tracks.items():
            best_iou = 0.0
            best_idx = -1

            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                score = iou(track["bbox"], det["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_idx >= 0 and best_iou > self.iou_thresh:
                det = detections[best_idx]
                updated_tracks[track_id] = {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "age": 0
                }
                assigned.add(best_idx)
            else:
                track["age"] += 1
                if track["age"] <= self.max_age:
                    updated_tracks[track_id] = track

        for i, det in enumerate(detections):
            if i not in assigned:
                updated_tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "age": 0
                }
                self.next_id += 1

        self.tracks = updated_tracks
        return self.get_active_tracks()

    def get_active_tracks(self):
        return [
            {
                "track_id": tid,
                "bbox": t["bbox"],
                "confidence": t["confidence"]
            }
            for tid, t in self.tracks.items()
        ]
