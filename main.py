from fastapi import FastAPI, UploadFile, File
import shutil
import os
import cv2
import uuid

from app.detector import BirdDetector
from app.tracker import Tracker
from app.weight import WeightEstimator
from app.utils import sample_frames, draw_overlay

app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/analyze_video")
def analyze_video(
    video: UploadFile = File(...),
    fps_sample: int = 5,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.3
):
    video_id = str(uuid.uuid4())
    video_path = f"{OUTPUT_DIR}/{video_id}.mp4"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    frames, timestamps, fps = sample_frames(video_path, fps_sample)

    detector = BirdDetector(conf_thresh)
    tracker = Tracker(iou_thresh)
    weight_estimator = WeightEstimator()

    counts = []
    annotated_frames = []

    for frame, ts in zip(frames, timestamps):
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        weight_estimator.update(tracks)

        counts.append({
            "timestamp": round(ts, 2),
            "count": len(tracks)
        })

        frame = draw_overlay(frame, tracks, len(tracks))
        annotated_frames.append(frame)

    # Save annotated video
    h, w, _ = annotated_frames[0].shape
    out_path = f"{OUTPUT_DIR}/annotated_{video_id}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_sample, (w, h)) # type: ignore

    for f in annotated_frames:
        out.write(f)
    out.release()

    weight_estimates = weight_estimator.estimate()

    response = {
        "counts": counts,
        "tracks_sample": list(weight_estimates.keys())[:5],
        "weight_estimates": {
            "unit": "relative_index",
            "per_bird": weight_estimates
        },
        "artifacts": {
            "annotated_video": out_path
        }
    }

    return response
