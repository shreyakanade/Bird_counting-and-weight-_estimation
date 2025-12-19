import cv2
import numpy as np
from ultralytics import YOLO

class PoultryProcessor:
    def __init__(self, model_path='yolov8n.pt'):
        # Using yolov8n.pt as a base; it will auto-download on first run
        self.model = YOLO(model_path)

    def process_frame(self, frame, conf_thresh):
        # Tracking birds (COCO class 14 is 'bird')
        # We use persist=True to maintain IDs across frames
        results = self.model.track(
            frame, 
            persist=True, 
            conf=conf_thresh, 
            iou=0.5, 
            classes=[14] 
        )
        
        annotated_frame = results[0].plot()
        detections = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                # Calculate Area Proxy: (Width * Height)
                area_proxy = (x2 - x1) * (y2 - y1)
                detections.append({
                    "id": int(obj_id),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "weight_index": float(area_proxy)
                })
        
        return annotated_frame, detections