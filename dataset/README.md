<div align="center">

<img src="https://img.shields.io/badge/Dataset-SAMVEDNA-blueviolet?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/Modalities-Audio%20%7C%20Video%20%7C%20AV-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Languages-English%20%7C%20Hindi%20%7C%20Punjabi-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Samples-24%2C300-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-Research%20Use-lightgrey?style=for-the-badge"/>

<br/><br/>

# 🎙️ SAMVEDNA Dataset
### *A Multimodal Multilingual Emotion Recognition Dataset*

**Speech · Facial Expressions · 6 Emotions · 3 Languages · 25 Actors**

---

[📥 Request Access](#-access--download) · [📄 Paper](#-citation) · [🔬 Use with EmoHuBERT](../EmoHuBERT) · [📊 Dataset Stats](#-dataset-statistics)

</div>

---


**SAMVEDNA** is a large-scale, **multimodal and multilingual emotion recognition dataset** designed to advance research in affective computing, speech processing, and human-computer interaction. The dataset captures emotional variation across **audio, video, and audio-visual modalities** from **25 professional actors** across **three Indian languages**.

---

## ✨ Key Highlights

| Feature | Details |
|---|---|
| 🎭 Emotion Classes | 6 (Happy, Sad, Anger, Disgust, Fear, Neutral) |
| 🌍 Languages | English, Hindi, Punjabi |
| 🎥 Modalities | Audio-only (.mp3), Video-only (.mp4), Audio-Visual (.mp4) |
| 👥 Actors | 25 (13 Male, 12 Female) |
| 🧾 Sentences | 18 unique tone-controlled sentences |
| 📦 Total Samples | **24,300 labeled files** |
| 🔖 Naming Convention | Systematic (Actor · Language · Emotion · Sentence) |
| 🎂 Age Group | 20–30 years |
| 📍 Region | North India (Punjab, Haryana, Himachal Pradesh, Uttarakhand, Chandigarh) |

---

## 🗂️ Dataset Structure

```
SAMVEDNA/
│
├── 📁 Data/
│   ├── 😠 Anger/
│   │   ├── A1_E_A_S1.mp3       ← Actor 1, English, Anger, Sentence 1 (audio)
│   │   ├── A1_E_A_S1.mp4       ← Actor 1, English, Anger, Sentence 1 (video/AV)
│   │   └── ...
│   ├── 🤢 Disgust/
│   ├── 😨 Fear/
│   ├── 😊 Happy/
│   ├── 😢 Sad/
│   └── 😐 Neutral/
│
├── 📄 README.md                 ← This file
├── 📄 DATASET_STRUCTURE.md     ← Folder and naming convention details
└── 📄 sample_metadata.csv      ← Metadata for all 24,300 samples
```

---

## 🔖 File Naming Convention

Every file follows a precise 4-part naming format:

```
[Actor]_[Language]_[Emotion]_[Sentence].[format]
```

**Example:** `A3_H_S_S12.mp3` = Actor 3 · Hindi · Sad · Sentence 12 · Audio

| Code | Meaning | Values |
|---|---|---|
| **Actor** | Actor identifier | `A1` – `A25` |
| **Language** | Language code | `E` = English, `H` = Hindi, `P` = Punjabi |
| **Emotion** | Emotion label | `H` Happy, `S` Sad, `A` Anger, `D` Disgust, `F` Fear, `N` Neutral |
| **Sentence** | Sentence number | `S1` – `S18` |
| **Format** | File type | `.mp3` (audio), `.mp4` (video or AV) |

---

## 🎭 Emotion Classes

| # | Emotion | Spoken Label | File Code |
|---|---|---|---|
| 0 | 😠 Anger | `Anger` | `A` |
| 1 | 🤢 Disgust | `Disgust` | `D` |
| 2 | 😨 Fear | `Fear` | `F` |
| 3 | 😊 Happy | `Happy` | `H` |
| 4 | 😢 Sad | `Sad` | `S` |
| 5 | 😐 Neutral | `Neutral` | `N` |

---

## 🌍 Language Coverage

| Language | Code | Type |
|---|---|---|
| English | `E` | Non-native / Global academic language |
| Hindi | `H` | Widely spoken Indian language |
| Punjabi | `P` | Regional language (North India) |

Each actor records all 18 sentences in all 3 languages across all 6 emotions, enabling **language-independent** and **cross-lingual** emotion recognition research.

---

## 📊 Dataset Statistics

```
Total Files Breakdown
─────────────────────────────────────────────────
  25 actors
× 18 sentences
×  6 emotions
×  3 languages
×  3 modalities (audio / video / audio-visual)
─────────────────────────────────────────────────
= 24,300 labeled samples
```

| Split | Files |
|---|---|
| Per Actor | 972 |
| Per Language | 8,100 |
| Per Emotion | 4,050 |
| Per Modality | 8,100 |
| **Total** | **24,300** |

---

## 🧪 Sample Sentences

The dataset uses 18 real-world sentences designed for practical, tone-sensitive scenarios:

> *"Can't you hear my voice?"*
> *"I tried to resolve this issue from my end."*
> *"I no longer want to use your services."*
> *"I have been waiting long to connect."*

These cover scenarios from **customer service**, **public office communication**, and **telephonic interaction** — making the dataset ideal for real-world deployment studies.

---

## 👥 Actor Demographics

| Attribute | Details |
|---|---|
| Total Actors | 25 |
| Male / Female | 13 / 12 |
| Age Range | 20–30 years |
| Training | Theatre and dramatics professionals |
| Region | North India |

Gender balance and regional diversity ensure cultural relevance and demographic fairness in trained models.

---

## 🚀 Applications

SAMVEDNA is suited for a wide range of research and applied tasks:

- 🤖 **Emotion-aware AI systems**
- 📞 **Customer support call analytics**
- 🧠 **Mental health monitoring tools**
- 💬 **Human-computer interaction (HCI)**
- 🌐 **Multilingual speech processing**
- 🔀 **Multimodal deep learning & fusion research**
- 📱 **Mobile emotion recognition applications**


## 📥 Access & Download

To access the dataset:
A **sample subset** (5 actors × 2 sentences × all emotions) is available for quick testing.

Sample Video can be downloaded from the below given link : 

https://drive.google.com/drive/folders/1mPk3-vFdVkGwlauo_X9FKqfGJuzMXONn?usp=drive_link

<div align="center">




</div>
