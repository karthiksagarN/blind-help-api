from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import traceback

app = FastAPI()
model = YOLO("yolov8x.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        results = model(image, conf=0.25)
        boxes = results[0].boxes

        if boxes is not None and boxes.cls.numel() > 0:
            detections = []
            for i in range(len(boxes.cls)):
                class_id = int(boxes.cls[i].item())
                class_name = model.names[class_id]
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]

                detections.append({
                    "label": class_name,
                    "confidence": round(confidence, 2),
                    "box": {
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1)
                    }
                })
            return JSONResponse(content={"detections": detections})

        return JSONResponse(content={"detections": []})

    except Exception as e:
        print("🔥 ERROR:")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)