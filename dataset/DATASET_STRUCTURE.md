# 📁 SAMVEDNA — Dataset Structure

## Full Folder Layout

```
SAMVEDNA/
│
└── Data/
    ├── Anger/
    │   ├── A1_E_A_S1.mp3
    │   ├── A1_E_A_S1.mp4
    │   ├── A1_H_A_S1.mp3
    │   ├── A1_H_A_S1.mp4
    │   └── ... (4,050 files total)
    │
    ├── Disgust/    → 4,050 files
    ├── Fear/       → 4,050 files
    ├── Happy/      → 4,050 files
    ├── Sad/        → 4,050 files
    └── Neutral/    → 4,050 files
```

## File Naming Convention

```
[Actor]_[Language]_[Emotion]_[Sentence].[ext]
```

### Encoding Reference

| Field    | Values                                      |
|----------|---------------------------------------------|
| Actor    | A1 – A25                                    |
| Language | E (English), H (Hindi), P (Punjabi)         |
| Emotion  | H (Happy), S (Sad), A (Anger), D (Disgust), F (Fear), N (Neutral) |
| Sentence | S1 – S18                                    |
| Format   | .mp3 (audio only), .mp4 (video / AV)        |

## Rules

1. Each emotion is in its own folder — names must match exactly: `Anger`, `Disgust`, `Fear`, `Happy`, `Sad`, `Neutral`
2. No nested subfolders inside emotion folders
3. File names are case-sensitive
4. Each `.mp3` file has a corresponding `.mp4` in the same folder

## Files Per Emotion Folder

```
25 actors × 18 sentences × 3 languages × 3 modalities = 4,050 files per emotion
6 emotions × 4,050 = 24,300 total files
```
