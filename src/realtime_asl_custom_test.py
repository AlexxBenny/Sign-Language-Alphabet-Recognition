# src/realtime_asl_test.py
import torch
import cv2
import mediapipe as mp
import numpy as np
from torchvision import models, transforms
from PIL import Image

# --- Load model ---
model_data = torch.load("../models/asl_mobilenetv2_finetuned_full.pth", map_location="cuda" if torch.cuda.is_available() else "cpu")
classes = model_data["classes"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(model_data["model_state"])
model.eval().to(device)

# --- Transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- Camera ---
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            # Get bounding box
            x_coords = [lm.x for lm in handLms.landmark]
            y_coords = [lm.y for lm in handLms.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
            xmin, ymin = max(0, xmin-20), max(0, ymin-20)
            xmax, ymax = min(w, xmax+20), min(h, ymax+20)
            hand_crop = frame[ymin:ymax, xmin:xmax]

           

            # inside while True:
            if hand_crop.size != 0:
                img = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)  # convert NumPy → PIL
                img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = model(img)
                    pred_class = classes[preds.argmax(dim=1).item()]
                cv2.putText(frame, f"{pred_class}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Realtime Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
