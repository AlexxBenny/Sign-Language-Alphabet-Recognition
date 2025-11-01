import cv2
import mediapipe as mp
import os
import re

# Base save directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "../data/custom_asl")
SAVE_DIR = os.path.abspath(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# Ask which letter to capture
label = input("Enter the letter you want to capture (A-Z): ").upper()
save_path = os.path.join(SAVE_DIR, label)
os.makedirs(save_path, exist_ok=True)

# --- Find the next available index ---
existing = [f for f in os.listdir(save_path) if f.lower().endswith(".jpg")]
if existing:
    # extract the last number from filenames like A_23.jpg
    nums = [int(re.findall(r"_(\d+)\.jpg", f)[0]) for f in existing if re.findall(r"_(\d+)\.jpg", f)]
    start_index = max(nums) + 1 if nums else 0
else:
    start_index = 0

count = start_index
print(f"Starting from {count} (existing images: {len(existing)})")

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Press 's' to save cropped hand, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20

            # Clamp within frame
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            clean_frame = frame.copy()
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            hand_crop = clean_frame[y_min:y_max, x_min:x_max]

            cv2.putText(frame, f"{label} ({count})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if hand_crop.size != 0:
                cv2.imshow("Hand Crop", cv2.resize(hand_crop, (200, 200)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                img_path = os.path.join(save_path, f"{label}_{count}.jpg")
                cv2.imwrite(img_path, hand_crop)
                print(f"Saved: {img_path}")
                count += 1
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                print(f"Saved {count - start_index} new images for {label}")
                exit()

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"Saved {count - start_index} new images for {label}")
