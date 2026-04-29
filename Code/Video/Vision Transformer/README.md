
The system recognizes **6 emotion categories**:
> `Anger` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad`

---

## 🧠 Model Architecture

| Component        | Details                                      |
|------------------|----------------------------------------------|
| Base Model       | `google/vit-base-patch16-224-in21k`          |
| Model Type       | Vision Transformer (ViT)                     |
| Input Resolution | 224 × 224 pixels                             |
| Pretrained On    | ImageNet-21k (14 million images)             |
| Fine-tuned On    | SAMVEDNA Video Emotion Dataset               |
| Output Classes   | 6 (Anger, Disgust, Fear, Happy, Neutral, Sad)|        |

---

## 📁 Project Structure

```
samvedna_emotions/
│
├── vit.py     ← Main source code (training + evaluation)
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
└── DATASET_STRUCTURE.md     ← How to organize your dataset
```

---

## ⚙️ Setup & Installation

### Step 1: Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA (recommended for training)
- pip package manager

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure dataset path
Open `vit.py` and update **line 76** with the path to your SAMVEDNA dataset:
```python
dataset_dir = Path(r'___________________')  # <-- change this
```

### Step 4: Run the code
```bash
python vit.py
```

---

## 📂 Dataset Structure

```
SAMVEDNA Dataset/
├── Anger/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── Disgust/
│   └── *.mp4
├── Fear/
│   └── *.mp4
├── Happy/
│   └── *.mp4
├── Neutral/
│   └── *.mp4
└── Sad/
    └── *.mp4
```

Each subfolder name must exactly match the emotion label. Videos must be `.mp4` format.

---

## 🔄 Pipeline Summary

```
Video Files (.mp4)
       ↓
Frame Extraction (every 30th frame via OpenCV)
       ↓
Image Transforms (Resize → Crop → Normalize)
       ↓
Frame Averaging → Single Image Representation
       ↓
ViT Model (Fine-tuned)
       ↓
Emotion Prediction (6 Classes)
       ↓
Evaluation: Confusion Matrix + Classification Report
```

---

## 🏋️ Training Configuration

| Hyperparameter              | Value                  |
|-----------------------------|------------------------|
| Learning Rate               | 2e-5                   |
| Train Batch Size            | 10                     |
| Eval Batch Size             | 4                      |
| Epochs                      | 20                     |
| Weight Decay                | 0.01                   |
| Train/Val Split             | 90% / 10%              |
| Best Model Selection        | Highest Accuracy       |
| Optimizer                   | AdamW (default)        |

---

## 📊 Output

After training, the script produces:
- **Model checkpoints** saved in `samvedna-vit-emotions/`
- **Training logs** in `logs/`
- **Confusion matrix** saved as `confusion_matrix.png`
- **Classification report** printed to console

---

## 📦 Dependencies

| Library          | Purpose                              |
|------------------|--------------------------------------|
| `torch`          | Deep learning framework              |
| `torchvision`    | Image transforms and utilities       |
| `transformers`   | ViT model and Trainer API            |
| `datasets`       | Dataset loading and splitting        |
| `opencv-python`  | Video frame extraction               |
| `Pillow`         | Image processing                     |
| `scikit-learn`   | Evaluation metrics                   |
| `matplotlib`     | Confusion matrix visualization       |
| `accelerate`     | Distributed training support         |

