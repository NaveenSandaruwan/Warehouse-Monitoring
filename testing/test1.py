from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (use a pre-trained model)
model = YOLO("yolov8n.pt")  # Replace with yolov8s.pt, yolov8m.pt, etc., for different sizes

# Load an image
image_path = "testing\example.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Perform object detection
results = model(image_path)

# Visualize the results on the image
annotated_image = results[0].plot()  # Draw bounding boxes and labels on the image

# Display the annotated image
cv2.imshow("YOLOv8 Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
