from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("videos/traffic.mp4")

if not cap.isOpened():
    print("Error: Video could not be opened")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output_video.mp4",
    fourcc,
    fps,
    (width, height)
)

# Create output video file
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output_video.mp4",
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("Error: Output video could not be created")
    cap.release()
    exit()

# Vehicle counters
vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

# IDs already counted
counted_ids = set()

# Counting line
line_y = 350

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # Draw counting line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        2
    )

    # Detect and track vehicles
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            center_y = (y1 + y2) // 2
            class_name = model.names[cls]

            # Count each tracked vehicle once
            if (
                class_name in vehicle_counts
                and center_y > line_y
                and track_id not in counted_ids
            ):
                vehicle_counts[class_name] += 1
                counted_ids.add(track_id)

            # Draw vehicle box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display vehicle name and ID
            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Draw black background for count panel
    cv2.rectangle(
        frame,
        (10, 10),
        (260, 155),
        (0, 0, 0),
        -1
    )

    # Display counts in yellow
    y = 40

    for name, count in vehicle_counts.items():
        cv2.putText(
            frame,
            f"{name.capitalize()}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        y += 30

    # Save the processed frame
    out.write(frame)

    # Display the video
    cv2.imshow("Vehicle Tracking and Counting System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Print final counts
print("\nFinal Vehicle Counts:")

for name, count in vehicle_counts.items():
    print(f"{name.capitalize()}: {count}")

# Close video resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nOutput video saved as output_video.mp4")