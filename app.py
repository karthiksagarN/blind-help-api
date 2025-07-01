# app.py
# Import necessary libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import traceback
import io

# Initialize the FastAPI app
app = FastAPI(title="YOLOv8 Object Detection API")

# Load the pretrained YOLOv8 model
# The model will be downloaded automatically on the first run
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    # If there's an error loading the model, log it.
    # This helps in debugging issues on the deployment server.
    print(f"🔥 ERROR loading YOLO model: {e}")
    traceback.print_exc()
    model = None

@app.get("/", summary="Root endpoint", description="A simple root endpoint to check if the API is running.")
async def read_root():
    """
    A simple endpoint to confirm that the API is up and running.
    """
    return {"message": "Welcome to the YOLOv8 Object Detection API!"}


@app.post("/detect/", summary="Detect Objects in an Image", description="Upload an image and get object detection results.")
async def detect_objects(file: UploadFile = File(...)):
    """
    This endpoint receives an image file, performs object detection using YOLOv8,
    and returns the detected objects with their labels, confidence scores, and bounding boxes.
    """
    # Check if the model was loaded successfully
    if model is None:
        return JSONResponse(content={"error": "Model could not be loaded. Please check server logs."}, status_code=500)

    try:
        # Read the contents of the uploaded file into memory
        contents = await file.read()
        
        # Convert the file contents to a NumPy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode the NumPy array into an image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if the image was decoded successfully
        if image is None:
            return JSONResponse(content={"error": "Failed to decode image. Please upload a valid image file."}, status_code=400)

        # Perform inference with the YOLO model
        # conf=0.25 means only detections with a confidence score > 25% will be returned
        results = model(image, conf=0.25)
        
        # Extract bounding boxes, class IDs, and confidence scores
        boxes = results[0].boxes

        detections = []
        # Check if any objects were detected
        if boxes is not None and boxes.cls.numel() > 0:
            # Iterate over each detected object
            for i in range(len(boxes.cls)):
                class_id = int(boxes.cls[i].item())
                class_name = model.names[class_id]
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()  # Bounding box coordinates [x1, y1, x2, y2]

                # Append the detection information to our list
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
        
        # Return the list of detections as a JSON response
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        # Print the full error traceback for debugging
        print("🔥 UNEXPECTED ERROR:")
        traceback.print_exc()
        # Return a generic server error response
        return JSONResponse(content={"error": "An internal server error occurred.", "detail": str(e)}, status_code=500)

# To run this app locally, use the command:
# uvicorn app:app --reload
