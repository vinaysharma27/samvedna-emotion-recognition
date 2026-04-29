
**6 Emotion Classes Supported:**

| # | Emotion | Label |
|---|---------|-------|
| 0 | 😠 Anger | `Anger` |
| 1 | 🤢 Disgust | `Disgust` |
| 2 | 😨 Fear | `Fear` |
| 3 | 😊 Happy | `Happy` |
| 4 | 😢 Sad | `Sad` |
| 5 | 😐 Neutral | `Neutral` |

---

## 🧠 Model Architecture

```
Audio Input (.mp3 / .wav / .ogg / .flac)
        ↓
MFCC Feature Extraction (librosa)
  - 40 MFCC coefficients
  - 40 Delta coefficients
  - 40 Delta-Delta coefficients
  = 120 features per time frame
        ↓
Sequence Input → (batch, time_steps, 120)
        ↓
┌─────────────────────────────────────┐
│  BiLSTM Layer 1                     │
│  hidden_size=256, bidirectional     │
│  → output: (batch, T, 512)          │
│  + Dropout(0.3)                     │
├─────────────────────────────────────┤
│  BiLSTM Layer 2                     │
│  hidden_size=256, bidirectional     │
│  → output: (batch, T, 512)          │
│  + Dropout(0.3)                     │
├─────────────────────────────────────┤
│  Temporal Mean Pooling              │
│  → (batch, 512)                     │
├─────────────────────────────────────┤
│  FC Layer: 512 → 128 (ReLU)         │
│  FC Layer: 128 → 6 (Softmax)        │
└─────────────────────────────────────┘
        ↓
Emotion Prediction (6 classes)
```

---

## 📁 Project Structure

```
bilstm-emotion-recognition/
│
├── 📄 README.md                          ← You are here
├── 📄 LICENSE                            ← Research-use license
├── 📄 requirements.txt                   ← Python dependencies
├── 🐍 bilstm_emotion_recognition.py      ← Full source code
│
└── 📁 docs/
    ├── DATASET_STRUCTURE.md              ← How to organize audio dataset
    └── HOW_TO_RUN.txt                    ← Step-by-step run guide
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/bilstm-emotion-recognition.git
cd bilstm-emotion-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Your Dataset Path
Open `bilstm_emotion_recognition.py` and update **line 57**:
```python
'dataset_path': r'E:\Emotion Datasets\SAMVEDNA Dataset\audio',  # ← change this
```

### Step 4: Run
```bash
python bilstm_emotion_recognition.py
```

---

## 📂 Expected Dataset Structure

```
audio/
├── Anger/
│   ├── anger_001.wav
│   ├── anger_002.mp3
│   └── ...
├── Disgust/   → audio files
├── Fear/      → audio files
├── Happy/     → audio files
├── Sad/       → audio files
└── Neutral/   → audio files
```

**Supported audio formats:** `.mp3` · `.wav` · `.ogg` · `.flac`

See [`docs/DATASET_STRUCTURE.md`](docs/DATASET_STRUCTURE.md) for full details.

---

## 🔄 Full Pipeline

```
1. Load audio files from emotion folders
         ↓
2. Stratified Train/Val/Test split (80/10/10)
         ↓
3. MFCC Feature Extraction (40 + delta + delta-delta = 120 features)
         ↓
4. EmotionAudioDataset + custom padding collate_fn
         ↓
5. Stacked Bi-LSTM training (50 epochs, early stopping patience=10)
         ↓
6. Evaluation: Accuracy, Classification Report, Confusion Matrix
         ↓
7. Per-emotion bar chart (Precision, Recall, F1)
         ↓
8. Inference function: predict single audio file
         ↓
9. Save model (.pt, .pth) + results (.json)
```

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Sample Rate | 16,000 Hz |
| MFCC Coefficients | 40 |
| Total Features | 120 (MFCC + Δ + ΔΔ) |
| Max Audio Duration | 10 seconds |
| Train / Val / Test Split | 80% / 10% / 10% |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Early Stopping Patience | 10 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5) |
| LSTM Hidden Size | 256 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Gradient Clipping | 1.0 |

---

## 📊 Output Files

After training, the following files are generated:

| File | Description |
|---|---|
| `bilstm_emotion_model.pt` | Best model state dict (saved during training) |
| `bilstm_emotion_model_full.pth` | Full model object (for easy loading) |
| `bilstm_results.json` | Test accuracy, loss, config, confusion matrix |
| `bilstm_training_history.png` | Loss & accuracy curves |
| `bilstm_confusion_matrix.png` | Confusion matrix heatmap |
| `bilstm_per_emotion_metrics.png` | Per-class Precision / Recall / F1 bar chart |

---

## 🔍 Inference — Predict a Single Audio File

```python
result = predict_emotion('path/to/audio.wav', model, CONFIG, CONFIG['device'])

print(result['predicted_emotion'])   # e.g. "Happy"
print(result['confidence'])          # e.g. 0.92
print(result['probabilities'])       # dict of all 6 class probabilities
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `torch` | Bi-LSTM model and training |
| `librosa` | MFCC audio feature extraction |
| `numpy` | Numerical array operations |
| `pandas` | Metrics dataframe |
| `scikit-learn` | Stratified split, confusion matrix, report |
| `matplotlib` | Training curves and metric plots |
| `seaborn` | Confusion matrix heatmap |
| `tqdm` | Training progress bars |
| `soundfile` | Audio file I/O support |

---
