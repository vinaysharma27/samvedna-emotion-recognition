# Audio Dataset Structure Guide

## Required Folder Layout

```
audio/
│
├── Anger/
│   ├── anger_001.wav
│   ├── anger_002.mp3
│   └── ...
│
├── Disgust/
│   └── audio files (.mp3/.wav/.ogg/.flac)
│
├── Fear/
│   └── audio files
│
├── Happy/
│   └── audio files
│
├── Sad/
│   └── audio files
│
└── Neutral/
    └── audio files
```

## Rules

1. Each emotion must be a **separate subfolder**
2. Subfolder names must match **exactly**: `Anger`, `Disgust`, `Fear`, `Happy`, `Sad`, `Neutral`
3. Supported audio formats: `.mp3`, `.wav`, `.ogg`, `.flac`
4. No nested subfolders inside emotion folders
5. Recommended minimum: **50+ audio files per emotion** for good training

## Update Dataset Path

Open `bilstm_emotion_recognition.py` and find **line 57**:

```python
'dataset_path': r'E:\Emotion Datasets\SAMVEDNA Dataset\audio',
```

Replace with your actual path:

| OS | Example |
|---|---|
| Windows | `r'C:\Users\Name\Desktop\audio'` |
| Linux/Mac | `'/home/name/data/audio'` |
| Google Colab | `'/content/drive/MyDrive/audio'` |

## Audio Format Tips

- **WAV** is preferred (lossless, best feature quality)
- **MP3** also works well
- Minimum recommended duration: **1–2 seconds** per clip
- Maximum processed duration: **10 seconds** (configurable via `max_duration` in CONFIG)
- Sample rate is automatically resampled to **16,000 Hz**
