import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
from PIL import Image
import os
import gc  # For garbage collection
import numpy as np
from torchvision.transforms import functional as F
from torch.optim import Adam
import torch.nn as nn
import math
import torchvision.transforms.functional as TF

# Set hyperparameters
IOU_THRESHOLD = 0.9
DATASET_IMAGES = "E:\\Masters2023-2025\\3rd Semester\\R&D Project\\NewScreenshot\\Port of Dover"
ANNOTATIONS_PATH = "C:\\Users\\HP\\Downloads\\cleaned_instances_default.json"
# Configurations
TRAIN_SPLIT = 0.95
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-3

# Step 1: Custom COCO Dataset Loader
class SAMDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Filter out invalid annotations
        valid_annotations = [
            ann for ann in annotations
            if "segmentation" in ann and ann["segmentation"] and isinstance(ann["segmentation"], list)
        ]

        if not valid_annotations:
            raise ValueError(f"No valid annotations for image ID {img_id}")

        # Load image
        img_path = os.path.join(self.image_dir, self.coco.loadImgs(img_id)[0]['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Create mask
        masks = np.zeros((image.height, image.width), dtype=np.uint8)
        for ann in valid_annotations:
            mask = self.coco.annToMask(ann)
            masks = np.maximum(masks, mask)

        # Pad the image and mask to make dimensions divisible by 32
        height, width = image.height, image.width
        pad_h = math.ceil(height / 32) * 32 - height
        pad_w = math.ceil(width / 32) * 32 - width
        padding = (0, 0, pad_w, pad_h)  # (left, top, right, bottom)

        image = TF.pad(image, padding)
        masks = TF.pad(Image.fromarray(masks), padding)

        # Apply transformations to the image (if any)
        if self.transforms:
            image = self.transforms(image)

        # Convert the mask to tensor (manually)
        masks = torch.tensor(np.array(masks), dtype=torch.float32)

        return image, masks

# Apply transformations
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 2: Load COCO Dataset
dataset = SAMDataset(DATASET_IMAGES, ANNOTATIONS_PATH, transforms=transformations)

# Split dataset
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Step 3: SAM Model Definition
class SAMModel(nn.Module):
    def __init__(self):
        super(SAMModel, self).__init__()
        # Define the layers of your model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # Add more layers as needed
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
model = SAMModel()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        # Debug input and mask shapes
        # print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")

        # Forward pass
        predictions = model(images)  # [batch_size, 1, h_out, w_out]
        # print(f"Predictions shape: {predictions.shape}")  # Debug model output shape

        # If output is downsampled too much, upsample back to mask size
        if predictions.shape[-2:] != masks.shape[-2:]:
            predictions = F.interpolate(predictions, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        # Ensure masks have the right shape
        if masks.dim() == 3:  # [batch_size, height, width]
            masks = masks.unsqueeze(1)  # [batch_size, 1, height, width]

        # Compute loss
        loss = criterion(predictions, masks.float())  # BCEWithLogitsLoss requires float masks

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            predictions = (predictions > 0.5).float()

            intersection = (predictions * masks).sum(dim=(1, 2, 3))
            union = (predictions + masks).clamp(0, 1).sum(dim=(1, 2, 3))
            iou = intersection / (union + 1e-6)
            iou_scores.extend(iou.cpu().numpy().tolist())

    mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0
    return mean_iou

losses = []
ious = []

# Step 5: Training and Validation
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_iou = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Validation IoU: {val_iou:.4f}")

    losses.append(train_loss)
    ious.append(val_iou)
    if val_iou >= IOU_THRESHOLD:
        print("IoU condition met. Stopping training.")
        break

# Now plot the loss and IoU values
epochs = range(1, len(losses) + 1)

plt.figure(figsize=(8, 6))

# Plot Loss
plt.plot(epochs, losses, label='Loss', color='blue', marker='o')

# Plot IoU on a secondary y-axis
plt.twinx()
plt.plot(epochs, ious, label='IoU', color='green', marker='o')

# Labels, legend, and title
plt.xlabel('Epochs')
plt.ylabel('IoU', color='green')
plt.title('Training Loss and Validation IoU vs Epochs')
plt.grid(True)

plt.legend(['Loss', 'IoU'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()

# Show the plots
plt.show()

# Step 6: Save Model
torch.save(model.state_dict(),  "C:\\Users\\HP\\Downloads\\fine_tuned_unetsamNew29.pth")
print("Model saved as C:\\Users\\HP\\Downloads\\fine_tuned_unet.pth")

# Show the plots