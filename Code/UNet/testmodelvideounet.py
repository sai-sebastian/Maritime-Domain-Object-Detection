import cv2
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from segmentation_models_pytorch import Unet  # Import U-Net from segmentation_models_pytorch
from scipy.ndimage import label as ndimage_label



MODEL_PATH = "C:\\Users\\HP\\Downloads\\fine_tuned_unet.pth"  # Path to your trained U-Net model
IMAGE_PATH = "E:\\Masters2023-2025\\3rd Semester\\R&D Project\\NewScreenshot\\CopyImages\\Dover_01_01_May_2024_Day (3).png"  # Path to your test image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained U-Net model
model = Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)  # Binary segmentation setup
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
model = model.half() if device.type == "cuda" else model


# Define transformations for the test image
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Use the same normalization as during training
])

# Define a function to extract bounding boxes and labels from the segmentation map
def extract_bounding_boxes(segmentation_map):
    labeled_map, num_features = ndimage_label(segmentation_map)
    bounding_boxes = []
    for region_id in range(1, num_features + 1):
        # Find pixels belonging to this region
        region_mask = (labeled_map == region_id).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the largest contour for bounding box calculation
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_boxes.append((x, y, x + w, y + h))  # (x_min, y_min, x_max, y_max)
    return bounding_boxes

# Process video and overlay bounding boxes with labels
def process_video_with_segmentation_optimized(video_path, model, transform, device, output_path=None):
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Optional: Resize frames to smaller dimensions for faster processing
    RESIZE_WIDTH, RESIZE_HEIGHT = 256, 256

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        original_size = frame_pil.size

        # Preprocess frame
        frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
        frame_tensor = frame_tensor.half() if device.type == "cuda" else frame_tensor

        # Predict segmentation map
        with torch.no_grad():
            output = model(frame_tensor)
            output = torch.sigmoid(output)
            prediction = (output > 0.5).float()
            prediction = torch.nn.functional.interpolate(
                prediction, size=original_size[::-1], mode="bilinear", align_corners=False
            )
            segmentation_map = prediction[0, 0].cpu().numpy()

        # Resize segmentation map to match the original frame size
        segmentation_map = cv2.resize(segmentation_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        # Create overlay visualization
        segmentation_overlay = (segmentation_map * 255).astype(np.uint8)
        segmentation_overlay = cv2.applyColorMap(segmentation_overlay, cv2.COLORMAP_JET)
        blended_frame = cv2.addWeighted(frame, 0.7, segmentation_overlay, 0.3, 0)

        # Display frame
        cv2.imshow("Segmentation", blended_frame)

        if output_path:
            video_writer.write(blended_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    if output_path:
        video_writer.release()
    cv2.destroyAllWindows()

# Example usage
VIDEO_PATH = "C:\\Users\\HP\\Downloads\\Recording 2025-01-27 231245.mp4"  # Path to your input video
OUTPUT_PATH = "output_video.mp4"  # Path to save the output video (optional)

process_video_with_segmentation_optimized(VIDEO_PATH, model, transform, device, output_path=OUTPUT_PATH)
