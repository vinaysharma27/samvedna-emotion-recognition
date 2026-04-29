

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
Wav2Vec2FeatureExtractor
  - Normalizes raw waveform
  - Resamples to 16,000 Hz
  - Pads / truncates to max_length (160,000 samples = 10s)
        ↓
HuBERT Encoder (facebook/hubert-base-superb)
  - 12 Transformer layers
  - 768 hidden size
  - 8 attention heads
  - CNN feature encoder (7 conv layers)
  - ~94M parameters
        ↓
┌─────────────────────────────────────┐
│  HuBERT Hidden States               │
│  → (batch, time_steps, 768)         │
│                                     │
│  Mean Pooling over time             │
│  → (batch, 768)                     │
│                                     │
│  Classification Head                │
│  FC Layer: 768 → 256 (ReLU)         │
│  FC Layer: 256 → 6 (Softmax)        │
└─────────────────────────────────────┘
        ↓
Emotion Prediction (6 classes)
```

---

## 📁 Project Structure

```
hubert-emotion-recognition/
│
├── 📄 README.md                          ← You are here
├── 📄 LICENSE                            ← Research-use license
├── 📄 requirements.txt                   ← Python dependencies
├── 🐍 hubert.py                          ← Full source code
│
└── 📁 docs/
    ├── DATASET_STRUCTURE.md              ← How to organize audio dataset
    └── HOW_TO_RUN.txt                    ← Step-by-step run guide
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hubert-emotion-recognition.git
cd hubert-emotion-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first run will automatically download `facebook/hubert-base-superb` (~360MB) from Hugging Face Hub. Requires internet connection on first run.

### Step 3: Set Your Dataset Path
Open `hubert.py` and update **line 38** inside `CONFIG`:
```python
'dataset_path': r'E:\Emotion Datasets\SAMVEDNA Dataset\audio',  # ← change this
```

### Step 4: Run
```bash
python hubert.py
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
3. Wav2Vec2FeatureExtractor preprocessing
   (normalize + pad/truncate to 10s @ 16kHz)
         ↓
4. EmotionAudioDataset + custom collate_fn
         ↓
5. HuBERT fine-tuning (20 epochs, early stopping patience=5)
         ↓
6. Evaluation: Accuracy, Classification Report, Confusion Matrix
         ↓
7. Per-emotion bar chart (Precision, Recall, F1)
         ↓
8. Inference function: predict single audio file
         ↓
9. Save model (HuggingFace format) + results (.json)
```

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Pretrained Model | `facebook/hubert-base-superb` |
| Sample Rate | 16,000 Hz |
| Max Audio Duration | 10 seconds (160,000 samples) |
| Train / Val / Test Split | 80% / 10% / 10% |
| Batch Size | 8 |
| Epochs | 20 (with early stopping) |
| Early Stopping Patience | 5 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| LR Scheduler | LinearLR (decay to 0.1×) |
| Gradient Clipping | 1.0 |
| Total Parameters | ~94M |

> **GPU Recommended.** Fine-tuning HuBERT on CPU is very slow. An NVIDIA GPU with ≥8GB VRAM is ideal. Reduce `batch_size` to 4 if you run out of memory.

---

## 📊 Output Files

After training, the following files are generated:

| File | Description |
|---|---|
| `hubert_emotion_model.pt` | Best model state dict (saved during training) |
| `./hubert_emotion_model/` | Full HuggingFace-format model directory |
| `hubert_results.json` | Test accuracy, loss, config, confusion matrix |
| `hubert_training_history.png` | Loss & accuracy curves |
| `hubert_confusion_matrix.png` | Confusion matrix heatmap |
| `hubert_per_emotion_metrics.png` | Per-class Precision / Recall / F1 bar chart |

---

## 🔍 Inference — Predict a Single Audio File

```python
result = predict_emotion('path/to/audio.wav', model, feature_extractor, CONFIG['device'])

print(result['predicted_emotion'])   # e.g. "Happy"
print(result['confidence'])          # e.g. 0.92
print(result['probabilities'])       # dict of all 6 class probabilities
```

### Reload a saved model for inference:
```python
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import torch

model = HubertForSequenceClassification.from_pretrained('./hubert_emotion_model')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./hubert_emotion_model')
model.eval()

result = predict_emotion('audio.wav', model, feature_extractor, torch.device('cpu'))
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `torch` | Model training and inference |
| `transformers` | HuBERT model + feature extractor |
| `librosa` | Audio loading and resampling |
| `numpy` | Numerical array operations |
| `pandas` | Metrics dataframe |
| `scikit-learn` | Stratified split, confusion matrix, report |
| `matplotlib` | Training curves and metric plots |
| `seaborn` | Confusion matrix heatmap |
| `tqdm` | Training progress bars |
| `soundfile` | Audio file I/O support |

