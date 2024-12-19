import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can replace 'yolov8n.pt' with other models like 'yolov8s.pt' for better accuracy

# Open the video file
cap = cv2.VideoCapture('testing/examplevideo.mp4')

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the frame width, height, and FPS (frames per second)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Set up the VideoWriter to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
output_path = 'testing/output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)  # Perform inference on the frame

    # Plot results on the frame (draw bounding boxes, labels, etc.)
    frame_with_results = results[0].plot()  # This will draw the bounding boxes on the frame

    # Write the processed frame to the output video
    out.write(frame_with_results)

    # Optionally, display the frame (for debugging purposes)
    cv2.imshow('Frame with Detection', frame_with_results)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to:", output_path)
