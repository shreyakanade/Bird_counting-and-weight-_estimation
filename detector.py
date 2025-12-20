from ultralytics import YOLO  # type: ignore

class BirdDetector:
    def __init__(self, conf_thresh=0.4):
        self.model = YOLO("yolov8n.pt")
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 14 and conf >= self.conf_thresh:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        return detections
