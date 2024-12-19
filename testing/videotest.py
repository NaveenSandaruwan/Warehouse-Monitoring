import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # You can replace 'yolov8n.pt' with the path to your YOLO model file

# Load the video
cap = cv2.VideoCapture("testing/examplevideo.mp4")

# Check if video is opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the frame width, height, and FPS
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Set up the VideoWriter to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 files
output_path = 'testing/output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv8
    results = model(frame)  # Detect objects in the current frame

    # Annotate the frame with bounding boxes and labels
    frame = results[0].plot()  # Use the first result for plotting

    # Optionally, display the frame (for debugging)
    cv2.imshow('Frame', frame)

    # Write the processed frame to the output video
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
