import numpy as np
from app.weight import WeightEstimator

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1) * (y2 - y1))


class WeightEstimator:
    def __init__(self):
        self.areas = {}

    def update(self, tracks):
        for t in tracks:
            tid = t["track_id"]
            area = bbox_area(t["bbox"])
            self.areas.setdefault(tid, []).append(area)

    def estimate(self):
        avg_areas = {tid: np.mean(a) for tid, a in self.areas.items()}
        if not avg_areas:
            return {}

        min_a = min(avg_areas.values())
        max_a = max(avg_areas.values())

        weights = {}
        for tid, a in avg_areas.items():
            if max_a > min_a:
                w = (a - min_a) / (max_a - min_a)
            else:
                w = 0.5
            weights[tid] = {
                "weight_index": round(float(w), 3),
                "confidence": 0.8
            }

        return weights
