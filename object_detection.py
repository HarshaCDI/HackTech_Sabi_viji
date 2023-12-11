from pathlib import Path

import cv2
import torch
import numpy as np
import pandas
from ultralytics import YOLO

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
weights = "/home/sabi/data/models/"
model = YOLO("yolov8x.pt")  # Download model if not already downloaded
model.to(device)
model.fuse()

# Define classes
classes = ["bus", "car", "motorcycle", "truck"]
CLASSES = [2]
CONFIDENCE = 0.5

# Define video source
video_path = Path("/home/sabi/misc/hackteck/phase3/final/vid.mp4").resolve()

# Initialize video capture
cap = cv2.VideoCapture(str(video_path))

while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if not success:
        print("Failure in reading frame")
        break

    # Preprocess frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make prediction
    results = model(frame)

    # Extract detections
    for result in results:
        for box in result.boxes:
            detections = box.xyxy[0]
            class_idx = int(box.cls[0].item())
            confidence = box.conf[0].item()

            if class_idx in CLASSES and confidence >= CONFIDENCE:
                # Get bounding boxes and labels
                # boxes = vehicles[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
                # labels = vehicles["name"].values

                xmin, ymin, xmax, ymax = detections.tolist()

                # Draw bounding boxes on frame
                cv2.rectangle(
                    frame,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    str(class_idx),
                    (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

    # Show frame with bounding boxes and labels
    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
