# Audio Dataset Structure 
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
5. Recommended minimum: **50+ audio files per emotion** for good fine-tuning

## Update Dataset Path

Open `hubert.py` and find the `CONFIG` dictionary (~line 38):

```python
'dataset_path': r'________________________________',
```

Replace with your actual path:

| OS | Example |
|---|---|
| Windows | `r'C:\Users\Name\Desktop\audio'` |
| Linux/Mac | `'/home/name/data/audio'` |
| Google Colab | `'/content/drive/MyDrive/audio'` |

## Audio Format Tips

- **WAV** is preferred (lossless, best quality)
- **MP3** also works well
- Minimum recommended duration: **1–2 seconds** per clip
- Maximum processed duration: **10 seconds** (160,000 samples at 16kHz)
  — configurable via `max_length` in CONFIG
- All audio is automatically resampled to **16,000 Hz** by the feature extractor

