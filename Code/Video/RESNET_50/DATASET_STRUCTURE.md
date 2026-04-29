# Dataset Structure Guide

## Required Folder Structure

```
Video/
│
├── Anger/
│   ├── anger_001.mp4
│   └── anger_002.mp4
│
├── Disgust/
│   └── *.mp4
│
├── Fear/
│   └── *.mp4
│
├── Happy/
│   └── *.mp4
│
├── Neutral/          
│   └── *.mp4
│
└── Sad/
    └── *.mp4
```

## Rules

1. Each emotion must be a **separate subfolder**.
2. Subfolder names must be **exactly**: `Anger`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`
3. Videos must be `.mp4` format.
4. No nested subfolders inside emotion folders.

## Updating the Path

Open `resnet.py` and find line 19:

```python
dataset_path = 'D:/All emotions/Video'
```

Replace with your actual path:
- Windows: `'C:\\Users\\YourName\\Desktop\\Video'`
- Linux/Mac: `'/home/yourname/data/Video'`
- Google Colab: `'/content/drive/MyDrive/Video'`


