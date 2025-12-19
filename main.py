import cv2
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Query
from processor import PoultryProcessor

app = FastAPI()
proc = PoultryProcessor()

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    fps_sample: int = 5,
    conf_thresh: float = 0.25
):
    job_id = str(uuid.uuid4())[:8]
    input_path = f"temp_{job_id}_{file.filename}"
    output_filename = f"result_{job_id}.mp4"
    output_path = os.path.join("outputs", output_filename)
    
    if not os.path.exists("outputs"): os.makedirs("outputs")

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    time_series = []
    tracks_sample = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Sample at specified FPS
        if frame_count % max(1, int(fps // fps_sample)) == 0:
            annotated_frame, detections = proc.process_frame(frame, conf_thresh)
            
            count = len(detections)
            timestamp = frame_count / fps
            
            time_series.append({"timestamp": round(timestamp, 2), "count": count})
            if detections and len(tracks_sample) < 5:
                tracks_sample.append(detections[0])
            
            out.write(annotated_frame)
        else:
            out.write(frame)
        
        frame_count += 1

    cap.release()
    out.release()
    os.remove(input_path)

    return {
        "counts": time_series,
        "tracks_sample": tracks_sample,
        "weight_estimates": {"unit": "pixel_area_index", "status": "Relative"},
        "artifacts": {"video_url": output_path}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)