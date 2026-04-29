
# The system recognizes **6 emotion categories**:
> `Anger` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad`

---

## 🧠 Model Architecture

| Component         | Details                                          |
|-------------------|--------------------------------------------------|
| Model             | ResNet-50 (custom implementation)                |
| Input Resolution  | 48 × 48 × 3 (RGB frames)                        |
| Output Classes    | 6 (Anger, Disgust, Fear, Happy, Neutral, Sad)    |
| Loss Function     | Categorical Cross-Entropy                        |
| Optimizer         | Adam                                             |
| Framework         | TensorFlow / Keras                               |

---

## 📁 Project Structure

```
samvedna6_emotions/
│
├── resnet.py    ← Main source code (training + evaluation)
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
├── DATASET_STRUCTURE.md     ← How to organize your dataset
└── HOW_TO_RUN.txt           ← Step-by-step run guide
```

---

## ⚙️ Setup & Installation

### Step 1: Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA (recommended)
- pip package manager

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure dataset path
Open `resnet.py` and update **line 19**:
```python
dataset_path = 'D:/All emotions/Video'  # <-- change to your path
```

### Step 4: Run the code
```bash
python vit.py
```

---

## 📂 Dataset Structure

```
Video/
├── Anger/      *.mp4
├── Disgust/    *.mp4
├── Fear/       *.mp4
├── Happy/      *.mp4
├── Neutral/    *.mp4
└── Sad/        *.mp4
```

Each subfolder name must exactly match the emotion label. Videos must be `.mp4` format.

---

## 🔄 Pipeline Summary

```
Video Files (.mp4)
       ↓
Frame Extraction (every 10th frame via OpenCV)
       ↓
Resize to 48×48 + Normalize [0,1]
       ↓
ResNet-50 Model (trained from scratch)
       ↓
6-Class Softmax Output
       ↓
Evaluation: Accuracy · Classification Report · Confusion Matrix
```

---

## 🏋️ Training Configuration

| Hyperparameter        | Value              |
|-----------------------|--------------------|
| Epochs                | 20                 |
| Batch Size            | 32                 |
| Train/Test Split      | 80% / 20%          |
| Frame Sampling        | Every 10th frame   |
| Input Size            | 48 × 48 × 3        |
| Number of Classes     | 6                  |

---

## 📊 Output Files

After training, the script produces:

| File                        | Description                          |
|-----------------------------|--------------------------------------|
| `emotion_recognition_model.h5` | Saved trained Keras model         |
| `training_history.xlsx`     | Epoch-wise accuracy & loss (Excel)   |
| `training_plot.png`         | Accuracy & loss curves               |
| `confusion_matrix.png`      | Confusion matrix heatmap             |
| `classification_report.txt` | Per-class precision, recall, F1      |

---

## 📦 Dependencies

| Library          | Purpose                              |
|------------------|--------------------------------------|
| `tensorflow`     | Deep learning framework (ResNet-50)  |
| `numpy`          | Numerical operations                 |
| `opencv-python`  | Video frame extraction               |
| `scikit-learn`   | Train/test split, metrics            |
| `matplotlib`     | Training curve plots                 |
| `pandas`         | Save training history to Excel       |
| `seaborn`        | Confusion matrix heatmap             |
| `openpyxl`       | Excel file writing support           |

---
