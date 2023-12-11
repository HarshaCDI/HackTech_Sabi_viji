from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse


def calculate_speed(track, current_ltrb, previous_ltrb, frame_rate):
    """
    Calculates the speed of a track based on its displacement and frame rate.
    Args:
        track: DeepSORT track object.
        current_ltrb: Bounding box coordinates of the current frame (xmin, ymin, xmax, ymax).
        previous_ltrb: Bounding box coordinates of the previous frame (xmin, ymin, xmax, ymax).
        frame_rate: Frame rate of the video.
    Returns:
        Speed in kilometers per hour.
    """
    if previous_ltrb is None:
        return 0  # No previous frame to compare, return 0

    # Calculate displacement in pixels
    displacement = abs(current_ltrb[0] - previous_ltrb[0])

    # Calculate the average of current and previous bounding box areas
    current_area = current_ltrb[2] * current_ltrb[3]
    previous_area = previous_ltrb[2] * previous_ltrb[3]
    avg_area = (current_area + previous_area) / 2.0

    # Define the width of a reference object in meters
    reference_object_width = (
        35886  # Replace with the actual width of the reference object in meters
    )

    # Convert displacement to meters based on the average bounding box area
    meters_per_pixel = reference_object_width / avg_area
    displacement_meters = displacement * meters_per_pixel

    # Convert displacement to meters per second based on frame rate
    speed_meters_per_second = displacement_meters / frame_rate

    # Convert speed to kilometers per hour
    speed_kilometers_per_hour = speed_meters_per_second * 3.6

    return speed_kilometers_per_hour


def main(video_path, frame_rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_filename = Path(video_path).stem

    # Define model
    weights = "/home/sabi/data/models/"
    model = YOLO("yolov8x.pt")  # Download model if not already downloaded
    model.to(device)
    model.fuse()

    # Define classes
    CLASSES = [2, 3, 5, 7]
    CONFIDENCE = 0.5

    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))

    # Initialize DeepSort tracker
    tracker = DeepSort(max_age=5)

    # Output file
    output_file_path = "output.txt"
    with open(output_file_path, "w") as output_file:
        output_file.write(
            "<test_video_name> <frame_no> <obj_id> <xmin> <ymin> <xmax> <ymax> <speed>\n"
        )

        previous_ltrb = None
        frame_no = 0

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Failure in reading frame")
                break

            frame_no += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(frame_rgb)

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

            tracks = tracker.update_tracks(detections, frame=frame_rgb)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                current_ltrb = track.to_ltrb()
                speed = calculate_speed(track, current_ltrb, previous_ltrb, frame_rate)
                previous_ltrb = current_ltrb

                # Write information to the output file
                output_file.write(
                    f"{video_filename} {frame_no} {track_id} {current_ltrb[0]} {current_ltrb[1]} "
                    f"{current_ltrb[2]} {current_ltrb[3]} {speed:.2f}\n"
                )

                # Draw bounding boxes on frame
                cv2.rectangle(
                    frame,
                    (int(current_ltrb[0]), int(current_ltrb[1])),
                    (int(current_ltrb[2]), int(current_ltrb[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (int(current_ltrb[0]), int(current_ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                if speed:
                    cv2.putText(
                        frame,
                        f"Speed: {speed:.2f} km/h",
                        (int(current_ltrb[0]), int(current_ltrb[1]) + 15),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSORT Video Tracker")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--frame_rate", type=float, required=True, help="Frame rate of the video"
    )
    args = parser.parse_args()

    main(video_path=args.video, frame_rate=args.frame_rate)
