# src/finetune_asl_full.py
"""
Fine-tune existing ASL model (asl_mobilenetv2.pth) on your custom hand signs (A–E)
without forgetting original Kaggle ASL knowledge.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset

# ---- CONFIG ----
BASE_MODEL_PATH = "../models/asl_mobilenetv2.pth"
CUSTOM_DATA_DIR = "../data/custom_asl"
KAGGLE_DATA_DIR = "../data/asl_alphabet_train/asl_alphabet_train"
SAVE_PATH = "../models/asl_mobilenetv2_finetuned_full.pth"

BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-5   # very small to prevent overwriting learned weights
IMG_SIZE = 128
# ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Kaggle ASL dataset (optional smaller subset to preserve memory) ---
kaggle_dataset = datasets.ImageFolder(KAGGLE_DATA_DIR, transform=transform)

# --- Load your custom dataset (A–E) ---
custom_dataset = datasets.ImageFolder(CUSTOM_DATA_DIR, transform=transform)

# --- Merge both ---
combined_dataset = ConcatDataset([kaggle_dataset, custom_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Load existing model ---
checkpoint = torch.load(BASE_MODEL_PATH, map_location=device)
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
classes = checkpoint["classes"]
print("Loaded model with classes:", len(classes))

# --- Fine-tuning setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting gentle fine-tuning...")
model.train()
for epoch in range(EPOCHS):
    total_loss, total_correct, total = 0, 0, 0
    for imgs, labels in combined_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss={total_loss/total:.4f}  Acc={total_correct/total:.4f}")

torch.save({"model_state": model.state_dict(), "classes": classes}, SAVE_PATH)
print("Saved improved model to:", SAVE_PATH)
