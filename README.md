Vehicle Tracking and Counting System

About the Project

This project detects, tracks, and counts vehicles in a traffic video using Python, OpenCV, and YOLO.

Features

- Detects cars, motorcycles, buses, and trucks.
- Tracks vehicles using unique tracking IDs.
- Displays vehicle counts on the video.
- Draws bounding boxes around detected vehicles.
- Saves the processed video as "output_video.mp4".

Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack

Project Structure

- "main.py" — Main Python program
- "videos/traffic.mp4" — Input traffic video
- "output_video.mp4" — Processed output video
- "yolo11n.pt" — YOLO model weights, if downloaded

How to Run

1. Install Python.
2. Install the required packages:
   "pip install ultralytics opencv-python"
3. Place your traffic video inside the "videos" folder and name it "traffic.mp4".
4. Run the program:
   "python main.py"

Output

The program displays vehicle detections, tracking IDs, and vehicle counts. It also saves the processed video as "output_video.mp4".

Future Improvements

- Improve vehicle-counting accuracy.
- Add vehicle counting in both directions.
- Display traffic statistics and charts.

Author

Student Project