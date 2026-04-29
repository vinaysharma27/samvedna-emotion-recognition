# Audio Dataset Structure Guide

## Required Folder Layout

```
audio/
│
├── Anger/
│   ├── sample_001.wav
│   ├── sample_002.mp3
│   └── ...
│
├── Disgust/   → audio files (.wav/.mp3/.ogg/.flac)
├── Fear/      → audio files
├── Happy/     → audio files
├── Neutral/   → audio files
└── Sad/       → audio files
```

## Rules

1. Each emotion must be a **separate subfolder**
2. Subfolder names must match **exactly**: `Anger`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`
3. Supported formats: `.wav`, `.mp3`, `.ogg`, `.flac`
4. No nested subfolders inside emotion folders
5. Recommended minimum: **50+ files per emotion** for good results

## Update the Dataset Path

Open `wav2vec2_emotion_recognition.py` and find **line 48**:

```python
DATA_PATH = r'E:\Emotion Datasets\SAMVEDNA Dataset\audio'
```

Replace with your actual path:

| OS | Example |
|---|---|
| Windows | `r'C:\Users\Name\Desktop\audio'` |
| Linux/Mac | `'/home/name/data/audio'` |
| Google Colab | `'/content/drive/MyDrive/audio'` |

## Output Directory

The script saves all results to:
```python
OUTPUT_DIR = "./wav2vec2_finetune_SAMVEDNA"
```
You can change this on **line 49** if needed.

## Audio Tips

- **WAV** format preferred (lossless)
- **MP3** works fine too
- Sample rate is automatically resampled to **16,000 Hz**
- Audio is padded or truncated to the model's maximum length
