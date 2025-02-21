import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from segmentation_models_pytorch import Unet  # Import U-Net from segmentation_models_pytorch
import cv2


# Define constants
MODEL_PATH = "C:\\Users\\HP\\Downloads\\fine_tuned_unet.pth"  # Path to your trained U-Net model
IMAGE_PATH = "E:\\Masters2023-2025\\3rd Semester\\R&D Project\\NewScreenshot\\CopyImages\\Dover_01_01_May_2024_Day (3).png"  # Path to your test image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained U-Net model
model = Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)  # Binary segmentation setup
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define transformations for the test image
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Use the same normalization as during training
])

# Load and preprocess the test image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Store original size for later resizing
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image, original_size

# Predict the segmentation mask
def predict_segmentation(model, image_tensor, original_size, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)  # Get the model's output
        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
        prediction = (output > 0.5).float()  # Binarize the predictions
    # Resize prediction back to original image size
    prediction = torch.nn.functional.interpolate(prediction, size=original_size[::-1], mode="bilinear", align_corners=False)
    return prediction[0, 0].cpu().numpy()

# Get bounding box from segmentation mask
def get_bounding_box(segmentation_map):
    contours, _ = cv2.findContours((segmentation_map * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h
    return None

# Load the test image
test_image_tensor, original_size = load_image(IMAGE_PATH)

# Generate the segmentation map
segmentation_map = predict_segmentation(model, test_image_tensor, original_size, device)

# Visualize the results
def visualize_results(image_path, segmentation_map):
    original_image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Segmentation map
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(segmentation_map, cmap="jet", alpha=0.5)  # Overlay segmentation map
    plt.title("Segmentation Map")
    plt.axis("off")


     # Get bounding box and add it to the plot
    bbox = get_bounding_box(segmentation_map)
    if bbox:
        x, y, w, h = bbox
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
        plt.text(x, y - 10, "Cruise Ship", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# Visualize the test image and segmentation result
visualize_results(IMAGE_PATH, segmentation_map)