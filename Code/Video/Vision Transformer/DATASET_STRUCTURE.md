## Required Folder Structure

Place your dataset anywhere on your machine. Then update the path inside `vit.py`.

```
SAMVEDNA Dataset/
│
├── Anger/
│   ├── anger_001.mp4
│   ├── anger_002.mp4
│   └── anger_003.mp4
│
├── Disgust/
│   ├── disgust_001.mp4
│   └── ...
│
├── Fear/
│   ├── fear_001.mp4
│   └── ...
│
├── Happy/
│   ├── happy_001.mp4
│   └── ...
│
├── Neutral/
│   ├── neutral_001.mp4
│   └── ...
│
└── Sad/
    ├── sad_001.mp4
    └── ...
```

## Rules

1. Each emotion must be a **separate subfolder**.
2. Subfolder names must be **exactly**: `Anger`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`
3. Videos must be in `.mp4` format (change extension in code if needed).
4. No nested subfolders inside emotion folders.

## Updating the Path

Open `vit.py` and find this line near the top:

```python
dataset_dir = Path(r'________________________________')
```

Replace it with the actual path on your system. Examples:

- Windows: `Path(r'C:\Users\YourName\Desktop\SAMVEDNA Dataset')`
- Linux/Mac: `Path('/home/yourname/data/SAMVEDNA_Dataset')`
- Google Colab: `Path('/content/drive/MyDrive/SAMVEDNA Dataset')`
