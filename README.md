# Real-Time ASL Alphabet Recognition System

A deep learning system for real-time American Sign Language alphabet recognition using computer vision and transfer learning.


## 🚀 Overview

This project implements a complete pipeline for real-time sign language recognition, from data collection to live inference. The system detects hand gestures via webcam and classifies them into A-Z letters using a fine-tuned MobileNetV2 model.

## ✨ Features

- **Real-time Hand Detection**: MediaPipe-based hand landmark detection
- **Deep Learning Model**: MobileNetV2 fine-tuned on ASL dataset
- **Custom Data Collection**: Built-in tool for capturing labeled hand images
- **Live Inference**: Real-time prediction with visual feedback
- **Transfer Learning**: Combines pre-trained knowledge with custom data

## 🛠️ Tech Stack

**Computer Vision**: OpenCV, MediaPipe Hands  
**Deep Learning**: PyTorch, MobileNetV2, Transfer Learning  
**Data Processing**: TorchVision, PIL, NumPy  
**Infrastructure**: Python, CUDA Support

## 📁 Project Structure
sign-language-recognition/
├── src/
│ ├── capture_custom_dataset.py # Data collection utility
│ ├── train_asl.py # Model training on ASL dataset
│ ├── finetune_asl_full.py # Fine-tuning with custom data
│ └── realtime_asl_custom_test.py # Real-time inference
├── models/ # Model weights directory
├── data/ # Dataset directory
├── requirements.txt
└── README.md


## ⚡ Quick StartUsage

### Installation
```bash
git clone https://github.com/alexxbenny/sign-language-alphabet-recognition
cd sign-languag-alphabet-recognition
pip install -r requirements.txt

## 🧠 Usage

### ▶️ Real-Time Prediction
Run the real-time sign language recognition system using your webcam:
```bash
python src/realtime_asl_custom_test.py
```

### 📸 Capture Custom Dataset
Capture your own hand signs to improve model accuracy:
```bash
python src/capture_custom_dataset.py
```
- Press **`s`** to save a frame  
- Press **`q`** to quit  
Images will be saved under `data/custom_asl/<LETTER>` folders.

### 🏋️ Train the Base Model
Train the model on the **Kaggle ASL dataset**:
```bash
python src/train_asl.py
```
This creates `models/asl_mobilenetv2.pth`.

### 🔧 Fine-Tune with Custom Data
Fine-tune the trained model on your **custom hand signs**:
```bash
python src/finetune_asl_full.py
```
This saves `models/asl_mobilenetv2_finetuned.pth`.

### 🔍 Test Fine-Tuned Model
Run real-time testing with your fine-tuned model:
```bash
python src/realtime_asl_custom_test.py
```

## 📊 Model Details

| Parameter | Description |
|------------|-------------|
| **Architecture** | MobileNetV2 (Transfer Learning) |
| **Input Size** | 128×128 RGB |
| **Output Classes** | 26 (A–Z) |
| **Training Data** | Kaggle ASL Alphabet Dataset + Custom Dataset |
| **Accuracy** | ~85–95% depending on fine-tuning |
| **Model Size** | ~9 MB (.pth) |

---

## ⚙️ Configuration

### 📂 Data Paths
| Type | Path |
|------|------|
| **Kaggle ASL Dataset** | `../data/asl_alphabet_train/asl_alphabet_train` |
| **Custom Dataset** | `../data/custom_asl` |
| **Saved Models** | `../models/` |

### 🔧 Training Hyperparameters
| Parameter | Value |
|------------|--------|
| **Batch Size** | 32–64 |
| **Learning Rate** | 1e-4 (base), 1e-5 (fine-tune) |
| **Image Size** | 128×128 |
| **Epochs** | 6–10 |
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |

---

## 🎯 Performance

| Metric | CPU | GPU (RTX 4050 Example) |
|---------|-----|-------------------------|
| **Inference Speed** | 15–20 FPS | 30–40 FPS |
| **Average Accuracy** | 85%+ | 90%+ |
| **Latency** | ~100ms | ~50ms |
| **Model Load Time** | <1s | <0.5s |


## 🧾 Model Weights

The trained model weights (`.pth` files) are excluded from version control due to file size limits.

You have two options to obtain them:

### 🏋️ Option 1: Train Your Own
Train the base model on the ASL dataset:
```bash
python src/train_asl.py
```

Then fine-tune it on your custom dataset:
```bash
python src/finetune_asl_full.py
```

### 📦 Option 2: Request Pretrained Weights
You can request pre-trained weights by contacting me.

📧 **Email:** alexbenny2004@gmail.com

---

### 💾 Model File Details

| File Name | Description |
|------------|-------------|
| `asl_mobilenetv2.pth` | Trained on full Kaggle ASL dataset |
| `asl_mobilenetv2_finetuned.pth` | Fine-tuned with your custom dataset |
| `asl_mobilenetv2_finetuned_full.pth` | Final refined model (A–Z stable) |

---

### ⚠️ Notes
- Models are saved automatically in the `models/` directory after training or fine-tuning.  
- If you train again, the latest model overwrites the previous one unless renamed manually.  
- GPU acceleration via CUDA is supported automatically if available.

## 🤝 Contributing

Contributions, ideas, and improvements are always welcome!  
Whether you’re fixing bugs, improving the model, or adding new features — every pull request helps.

### 🪜 Steps to Contribute

1. **Fork the Repository**
   ```bash
   git fork https://github.com/alexxbenny/sign-language-alphabet-recognition.git
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/alexxbenny/sign-language-alphabet-recognition.git
   cd sign-language-recognition
   ```

3. **Create a New Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Improve code, fix issues, or add new features.  
   - Follow consistent naming conventions and code formatting.

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: your feature or fix description"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to your fork on GitHub.  
   - Click **"Compare & Pull Request"**.  
   - Describe your changes clearly.  

---

### 🧩 Contribution Guidelines

- Follow **PEP8** coding standards.  
- Keep commits **atomic** and **well-documented**.  
- Add comments for any complex logic.  
- Use descriptive variable and function names.  
- Avoid committing large datasets or model files.

---

🧡 *Every small contribution helps make real-time sign recognition more accessible to everyone.*

## 📜 License

This project is licensed under the **MIT License** — meaning you’re free to use, modify, and distribute it, provided proper credit is given.

### 🧾 License Summary

```
MIT License

Copyright (c) 2025 Alex Benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

### 💡 In Simple Terms
You can:
- ✅ Use this project in your own work  
- ✅ Modify and improve it  
- ✅ Share or publish your changes  
- ❌ Not claim the original code as your own  

---

Note: This is a proof-of-concept project demonstrating real-time computer vision and transfer learning techniques. For production use, additional optimization and validation is recommended.

