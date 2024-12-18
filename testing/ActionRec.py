import torch
import torchvision.transforms.functional as F
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.data.encoded_video import EncodedVideo
import requests
import cv2
import numpy as np

# Load the pre-trained SlowFast model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = slowfast_r50(pretrained=True).to(device)
model.eval()

# Download the class labels for action recognition (Kinetics-400 dataset)
label_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
response = requests.get(label_url)
if response.status_code == 200:
    kinetics_labels = response.text.strip().split("\n")
else:
    print("Error: Could not download the Kinetics-400 labels.")
    exit()

# Video processing function
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = F.to_pil_image(frame)
    frame = F.resize(frame, size=[256, 256])
    frame = F.center_crop(frame, output_size=[224, 224])
    frame = F.to_tensor(frame)
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

# Function to predict action
def predict_action(frames):
    frames = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)  # Convert to (B, C, T, H, W)
    slow_pathway = frames
    fast_pathway = torch.index_select(frames, 2, torch.linspace(0, frames.shape[2] - 1, frames.shape[2] * 4).long())
    inputs = [slow_pathway, fast_pathway]
    with torch.no_grad():
        preds = model(inputs)
    preds = torch.nn.functional.softmax(preds, dim=1)
    top_pred = torch.argmax(preds, dim=1).item()
    return kinetics_labels[top_pred]

# Real-time video processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    buffer_size = 32  # Number of frames to process at a time
    action = "Detecting..."  # Initialize action variable

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(preprocess_frame(frame))

        if len(frame_buffer) == buffer_size:
            action = predict_action(frame_buffer)
            frame_buffer = []

        cv2.putText(frame, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = r"testing\Actions.mp4"  # Use raw string to avoid invalid escape sequences
process_video(video_path)