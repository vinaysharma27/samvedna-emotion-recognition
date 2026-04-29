

**6 Emotion Classes:**

| # | Emotion | Label |
|---|---------|-------|
| 0 | 😠 Anger | `Anger` |
| 1 | 🤢 Disgust | `Disgust` |
| 2 | 😨 Fear | `Fear` |
| 3 | 😊 Happy | `Happy` |
| 4 | 😐 Neutral | `Neutral` |
| 5 | 😢 Sad | `Sad` |

---

## 🧠 Model Architecture

```
Audio Input (.mp3 / .wav / .ogg / .flac)
        ↓
Wav2Vec2FeatureExtractor
  - Resample to 16,000 Hz
  - Normalize raw waveform
  - Pad/truncate with attention mask
        ↓
┌──────────────────────────────────────────────────┐
│  facebook/wav2vec2-base                          │
│  (Pretrained Self-Supervised Speech Model)       │
│                                                  │
│  Feature Extractor (CNN):        [FROZEN ep1-2] │
│    7 × Conv1D layers                             │
│    → local acoustic representations              │
│                                                  │
│  Transformer Encoder:           [ALWAYS TRAINED] │
│    12 layers × 768 hidden dims                   │
│    → contextual speech representations           │
│                                                  │
│  Classification Head:           [ALWAYS TRAINED] │
│    Mean pooling over time                        │
│    → Linear(256 → 6)                             │
└──────────────────────────────────────────────────┘
        ↓
Softmax → 6-class Emotion Prediction
```

---

## 📁 Project Structure

```
wav2vec2-emotion-recognition/
│
├── 📄 README.md                           ← You are here
├── 📄 LICENSE                             ← Research-use license
├── 📄 requirements.txt                    ← Python dependencies
├── 🐍 wav2vec2_emotion_recognition.py     ← Full source code (16 blocks)
│
└── 📁 docs/
    ├── DATASET_STRUCTURE.md               ← How to organize audio dataset
    └── HOW_TO_RUN.txt                     ← Step-by-step run guide
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/wav2vec2-emotion-recognition.git
cd wav2vec2-emotion-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Your Dataset Path
Open `wav2vec2_emotion_recognition.py` and update **line 48**:
```python
DATA_PATH = r'E:\Emotion Datasets\SAMVEDNA Dataset\audio'  # ← change this
```

### Step 4: Run
```bash
python wav2vec2_emotion_recognition.py
```

> **Note:** On first run, Wav2Vec2 model weights (~360 MB) are automatically downloaded from HuggingFace. Requires internet connection.

---

## 📂 Expected Dataset Structure

```
audio/
├── Anger/     → .wav / .mp3 / .ogg / .flac files
├── Disgust/   → audio files
├── Fear/      → audio files
├── Happy/     → audio files
├── Neutral/   → audio files
└── Sad/       → audio files
```

Subfolder names must match **exactly**. See [`docs/DATASET_STRUCTURE.md`](docs/DATASET_STRUCTURE.md) for full details.

---

## 🔄 Full Pipeline (16 Blocks)

```
Block 1–2:  Imports, reproducibility (SEED=42), device setup
       ↓
Block 3:    Paths, hyperparameters configuration
       ↓
Block 4–5:  AudioDataset class + collate_fn (with attention mask padding)
       ↓
Block 6:    Auto-create DataFrame from folder structure → train/val split (80/20)
       ↓
Block 7:    Load Wav2Vec2FeatureExtractor + label mappings
       ↓
Block 8:    Create DataLoaders (train & val)
       ↓
Block 9:    Initialize Wav2Vec2ForSequenceClassification
            → Freeze CNN feature extractor (Phase 1)
       ↓
Block 10:   AdamW optimizer + LinearWarmup scheduler
       ↓
Block 11:   train_one_epoch() + evaluate() functions
       ↓
Block 12:   Training loop (up to 50 epochs)
            → Unfreeze CNN at Epoch 3 (Phase 2)
            → Early stopping (patience=5)
            → Save best_model.pth on val acc improvement
       ↓
Block 13:   Plot training curves → training_curves.png
       ↓
Block 14:   Load best model → confusion matrix + classification report
       ↓
Block 15:   Save training_history.csv
       ↓
Block 16:   Save full model (HuggingFace format) + run inference example
            → Summary printout
```

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Pretrained Model | `facebook/wav2vec2-base` |
| Sample Rate | 16,000 Hz |
| Train / Val Split | 80% / 20% (stratified) |
| Batch Size | 8 |
| Max Epochs | 50 |
| Early Stopping Patience | 5 epochs |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| LR Scheduler | Linear warmup (200 steps) → linear decay |
| Gradient Clipping | 1.0 |
| Freeze Strategy | CNN frozen epochs 1–2, unfrozen from epoch 3 |
| Random Seed | 42 |

---

## 📊 Output Files

All outputs are saved to `./wav2vec2_finetune_SAMVEDNA/`:

| File | Description |
|---|---|
| `best_model.pth` | Best model state dict (saved on val accuracy improvement) |
| `finetuned_wav2vec2_model_SAMVEDNA/` | Full HuggingFace model + feature extractor |
| `training_curves.png` | Loss & accuracy curves across epochs |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `training_history.csv` | Epoch-by-epoch train/val loss & accuracy |

---

## 🔍 Inference — Predict a Single Audio File

```python
emotion, confidence = predict_emotion(
    'path/to/audio.wav',
    inference_model,
    inference_feature_extractor_obj,
    DEVICE,
    id2label
)

print(emotion)      # e.g. "Happy"
print(confidence)   # e.g. 0.94
```

---

## 🔁 Reload Saved Model

```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    './wav2vec2_finetune_SAMVEDNA/finetuned_wav2vec2_model_SAMVEDNA'
)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    './wav2vec2_finetune_SAMVEDNA/finetuned_wav2vec2_model_SAMVEDNA'
)
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `transformers` | Wav2Vec2 model & feature extractor |
| `librosa` | Audio loading & resampling |
| `numpy` | Numerical operations |
| `pandas` | DataFrame management & CSV export |
| `scikit-learn` | Stratified split, confusion matrix, report |
| `matplotlib` | Training curve & confusion matrix plots |
| `tqdm` | Training progress bars |
| `soundfile` | Audio file I/O |
| `accelerate` | Efficient model loading |

---
