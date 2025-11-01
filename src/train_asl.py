# src/train_asl.py
"""
Train a classifier on the ASL alphabet image dataset using transfer learning.
Saves model to models/asl_mobilenetv2.pth
"""

import os
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

# --------- CONFIG ----------
DATA_DIR = "../data/asl_alphabet_train/asl_alphabet_train"   # relative to src/
SAVE_PATH = "../models/asl_mobilenetv2.pth"
IMG_SIZE = 128            # resize images to 128x128 (balance quality & speed)
BATCH_SIZE = 64
EPOCHS = 6                # start small; increase if accuracy still low
LR = 1e-4                 # learning rate
NUM_WORKERS = 0           # for DataLoader parallelism; set lower on low-core machines
# --------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 1) transforms: what we do to input images before feeding model
# Resize -> ToTensor (scales pixels 0-255 -> 0.0-1.0) -> Normalize (ImageNet mean/std)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (for pretrained)
                         std=[0.229, 0.224, 0.225])
])

# 2) ImageFolder will map subfolder names to numeric labels
dataset = datasets.ImageFolder(os.path.join(os.path.dirname(__file__), DATA_DIR), transform=transform)
print(f"Total images: {len(dataset)}")
num_classes = len(dataset.classes)
print("Classes:", num_classes, dataset.classes[:10])

# 3) Split dataset into train / val (e.g., 90% train, 10% val)
val_pct = 0.1
val_size = int(len(dataset) * val_pct)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 4) Model: MobileNetV2 (lightweight, fast) pretrained on ImageNet
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# Replace classifier to match our number of classes
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# 5) Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

if __name__ == "__main__":
    # 6) Training loop (simple)
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS}  TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.4f}  ValAcc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes": dataset.classes
            }, os.path.join(os.path.dirname(__file__), SAVE_PATH))
            print("Saved best model", SAVE_PATH)

    print("Training finished. Best val acc:", best_val_acc)
