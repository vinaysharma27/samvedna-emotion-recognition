

import gc
gc.collect()  # Free memory

# ==============================================================================
# SECTION 1: IMPORTS & DATASET LOADING
# ==============================================================================

from datasets import Dataset
from pathlib import Path
import os
import random
import shutil
import torch

# Check GPU availability
print("GPU Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset_dir = Path(r'________________________')  # <-- UPDATE THIS PATH

# Emotion categories (must match folder names exactly)
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Load video file paths and integer labels
video_paths = []
labels = []

for label, emotion in enumerate(emotions):
    emotion_dir = dataset_dir / emotion
    for video in os.listdir(emotion_dir):
        if video.endswith('.mp4'):  # Adjust extension if your videos use a different format
            video_paths.append(str(emotion_dir / video))
            labels.append(label)

# Build HuggingFace Dataset object
dataset = Dataset.from_dict({'video': video_paths, 'label': labels})

# Train/Validation split: 90% train, 10% validation
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds   = splits['test']

print(f"Training set size:   {len(train_ds)}")
print(f"Validation set size: {len(val_ds)}")

# Label mappings (integer index <-> emotion name)
id2label = {id: label for id, label in enumerate(emotions)}
label2id = {label: id for id, label in id2label.items()}

print(f"Example label for sample 0: {id2label[train_ds[0]['label']]}")


# ==============================================================================
# SECTION 2: VIDEO FRAME EXTRACTION & DATA TRANSFORMS
# ==============================================================================

import cv2
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)
from torch.utils.data import DataLoader
import numpy as np

# ImageNet normalization statistics (standard for ViT pretrained on ImageNet)
image_mean = [0.485, 0.456, 0.406]
image_std  = [0.229, 0.224, 0.225]
size = 224  # ViT input resolution

normalize = Normalize(mean=image_mean, std=image_std)

# Training transforms: random crop + flip for augmentation
_train_transforms = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

# Validation transforms: deterministic resize + center crop
_val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])


def extract_frames(video_path, frame_rate=30):
    """
    Extract frames from a video file, keeping 1 frame every 'frame_rate' frames.

    Args:
        video_path (str): Path to the .mp4 video file.
        frame_rate (int): Interval for frame sampling (default: every 30th frame).

    Returns:
        list[PIL.Image]: List of extracted frames as PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    # Convert BGR (OpenCV) to RGB PIL Images
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    return frames


def train_transforms(examples):
    """Apply training augmentation transforms to a batch of video paths."""
    video_frames = [extract_frames(vp) for vp in examples['video']]
    examples['pixel_values'] = [
        [_train_transforms(frame.convert("RGB")) for frame in frames]
        for frames in video_frames
    ]
    return examples


def val_transforms(examples):
    """Apply validation transforms to a batch of video paths."""
    video_frames = [extract_frames(vp) for vp in examples['video']]
    examples['pixel_values'] = [
        [_val_transforms(frame.convert("RGB")) for frame in frames]
        for frames in video_frames
    ]
    return examples


# Register transforms with the datasets
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)


# ==============================================================================
# SECTION 3: DATA COLLATION (batching frames into tensors)
# ==============================================================================

import torchvision.transforms as T

max_frames   = 100          # Maximum frames per video to process
target_size  = (224, 224)   # Consistent spatial resolution for ViT

resize_transform = T.Resize(target_size)


def collate_fn(examples):
    """
    Custom collate function for DataLoader.
    Extracts frames, resizes them, averages across the time dimension,
    and stacks into a batch tensor of shape (batch_size, C, H, W).

    Args:
        examples (list[dict]): Each dict has 'pixel_values' (list of tensors) and 'label'.

    Returns:
        dict: {'pixel_values': Tensor(B, C, H, W), 'labels': Tensor(B,)}
    """
    pixel_values = []
    labels = []

    for example in examples:
        frames = example["pixel_values"]

        # Ensure all frames are torch Tensors
        frames = [
            frame if isinstance(frame, torch.Tensor) else ToTensor()(frame)
            for frame in frames
        ]

        # Resize to consistent spatial dimensions
        frames = [resize_transform(frame) for frame in frames]

        # Stack frames: (num_frames, C, H, W) → average → (C, H, W)
        frames = torch.stack(frames)
        frames = frames.mean(dim=0)

        pixel_values.append(frames)
        labels.append(example["label"])

    pixel_values = torch.stack(pixel_values)  # (B, C, H, W)
    labels       = torch.tensor(labels)       # (B,)

    return {"pixel_values": pixel_values, "labels": labels}


# Quick sanity check: verify batch shapes
train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
batch = next(iter(train_dataloader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"Batch key: {k}, shape: {v.shape}")


# ==============================================================================
# SECTION 4: MODEL DEFINITION & TRAINING
# ==============================================================================

from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# Load pretrained ViT image processor and model
# Model: google/vit-base-patch16-224-in21k (pretrained on ImageNet-21k)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    id2label=id2label,
    label2id=label2id
)


def compute_metrics(eval_pred):
    """Compute accuracy score for evaluation during training."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


# Training hyperparameters
args = TrainingArguments(
    output_dir="samvedna-vit-emotions",    # Directory to save model checkpoints
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# Train the model
print("\n[INFO] Starting training...")
trainer.train()
print("[INFO] Training complete.")


# ==============================================================================
# SECTION 5: EVALUATION — Confusion Matrix & Classification Report
# ==============================================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

print("\n[INFO] Running predictions on validation set...")
outputs = trainer.predict(val_ds)

y_true = outputs.label_ids
y_pred = np.argmax(outputs.predictions, axis=1)

# Retrieve label names from the model config
if hasattr(model.config, 'id2label'):
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
else:
    label_names = [str(i) for i in range(len(set(y_true)))]

# Plot Confusion Matrix
cm   = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix — Samvedna eMOTIONS (ViT)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("[INFO] Confusion matrix saved to confusion_matrix.png")

# Print Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=label_names))

# Print Trainer evaluation metrics
print("="*60)
print("TRAINER EVALUATION METRICS")
print("="*60)
print(outputs.metrics)
