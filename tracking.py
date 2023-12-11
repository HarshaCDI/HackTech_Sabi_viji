from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

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

# Initialize DeepSort tracker
tracker = DeepSort(max_age=5)

while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    if not success:
        print("Failure in reading frame")
        break

    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make prediction
    results = model(frame_rgb)

    # Extract detections
    detections = []
    for result in results:
        for box in result.boxes:
            class_idx = int(box.cls[0].item())
            confidence = box.conf[0].item()

            if class_idx in CLASSES and confidence >= CONFIDENCE:
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                detections.append(
                    (
                        (xmin, ymin, xmax - xmin, ymax - ymin),
                        confidence,
                        class_idx,
                    )
                )

    # Update tracks with DeepSort
    tracks = tracker.update_tracks(detections, frame=frame_rgb)

    # Draw bounding boxes on frame with tracking information
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

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
            f"ID: {track_id}",
            (int(xmin), int(ymin) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Show frame with bounding boxes and labels
    cv2.imshow("Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
