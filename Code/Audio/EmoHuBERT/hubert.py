import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
warnings.filterwarnings('ignore')


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 42
set_seed(SEED)

CONFIG = {
    'random_seed': SEED,
    'dataset_path': r'E:\Emotion Datasets\SAMVEDNA Dataset\audio',
    'sr': 16000,
    'max_duration': 10.0,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'batch_size': 8,
    'epochs': 20,
    'learning_rate': 2e-5,
    'max_length': 160000,
    'hubert_model': 'facebook/hubert-base-superb',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

print('✓ Configuration loaded')
print(f'  Device: {CONFIG["device"]}')
print(f'  Emotions: {EMOTIONS}')


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_audio_data(dataset_path, emotions=EMOTIONS, sr=16000, max_duration=10.0):
    """Load audio files from emotion folders."""
    file_paths = []
    stats = defaultdict(int)
    errors = []
    
    for emotion in emotions:
        emotion_dir = os.path.join(dataset_path, emotion)
        
        if not os.path.exists(emotion_dir):
            print(f'⚠ Warning: {emotion_dir} not found')
            continue
        
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.ogg', '*.flac']:
            audio_files.extend(Path(emotion_dir).glob(ext))
        
        for audio_path in audio_files:
            try:
                y, _ = librosa.load(str(audio_path), sr=sr, duration=max_duration)
                if len(y) > 0:
                    file_paths.append((str(audio_path), emotion))
                    stats[emotion] += 1
            except Exception as e:
                errors.append((str(audio_path), str(e)))
    
    print('\n✓ Dataset loaded successfully')
    print('Samples per emotion:')
    for emotion in emotions:
        count = stats.get(emotion, 0)
        print(f'  {emotion}: {count}')
    print(f'Total samples: {sum(stats.values())}')
    
    if errors:
        print(f'⚠ {len(errors)} files had errors')
    
    return file_paths, dict(stats)

print('Loading dataset...')
file_paths, stats = load_audio_data(CONFIG['dataset_path'])


# ============================================================================
# 3. STRATIFIED SPLIT
# ============================================================================

def stratified_split(file_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Stratified split maintaining emotion distribution."""
    emotion_groups = defaultdict(list)
    for path, emotion in file_paths:
        emotion_groups[emotion].append((path, emotion))
    
    train_set, val_set, test_set = [], [], []
    np.random.seed(seed)
    
    for emotion, samples in emotion_groups.items():
        np.random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        train_set.extend(samples[:n_train])
        val_set.extend(samples[n_train:n_train+n_val])
        test_set.extend(samples[n_train+n_val:])
    
    return train_set, val_set, test_set

train_set, val_set, test_set = stratified_split(file_paths, seed=SEED)

print('✓ Data split completed')
print(f'  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')


# ============================================================================
# 4. DATASET CLASS
# ============================================================================

class EmotionAudioDataset(Dataset):
    def __init__(self, file_paths, feature_extractor, sr=16000, max_length=160000):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor
        self.sr = sr
        self.max_length = max_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path, emotion = self.file_paths[idx]
        
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            if len(y) < self.max_length:
                y = np.pad(y, (0, self.max_length - len(y)), mode='constant')
            else:
                y = y[:self.max_length]
            
            inputs = self.feature_extractor(
                y, sampling_rate=self.sr, return_tensors='pt', padding=True
            )
            
            return {
                'input_values': inputs['input_values'].squeeze(),
                'labels': torch.tensor(EMOTION_TO_IDX[emotion], dtype=torch.long),
                'audio_path': audio_path,
                'emotion': emotion
            }
        except Exception as e:
            print(f'Error loading {audio_path}: {e}')
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_values': input_values,
        'labels': labels
    }


# ============================================================================
# 5. LOAD MODEL
# ============================================================================

print(f'\nLoading HuBERT model: {CONFIG["hubert_model"]}')

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG['hubert_model'])

model = HubertForSequenceClassification.from_pretrained(
    CONFIG['hubert_model'],
    num_labels=len(EMOTIONS),
    id2label=IDX_TO_EMOTION,
    label2id=EMOTION_TO_IDX
)

model = model.to(CONFIG['device'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('✓ Model loaded')
print(f'  Total parameters: {total_params:,}')
print(f'  Trainable parameters: {trainable_params:,}')




# ============================================================================
# 6. CREATE DATA LOADERS
# ============================================================================

train_dataset = EmotionAudioDataset(train_set, feature_extractor, sr=CONFIG['sr'])
val_dataset = EmotionAudioDataset(val_set, feature_extractor, sr=CONFIG['sr'])
test_dataset = EmotionAudioDataset(test_set, feature_extractor, sr=CONFIG['sr'])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    collate_fn=collate_fn
)

print('✓ Data loaders created')
print(f'  Train batches: {len(train_loader)}')
print(f'  Val batches: {len(val_loader)}')
print(f'  Test batches: {len(test_loader)}')



# ============================================================================
# 7. TRAINING SETUP
# ============================================================================

optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
criterion = nn.CrossEntropyLoss()

total_steps = len(train_loader) * CONFIG['epochs']
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

print('✓ Optimizer and scheduler configured')



# ============================================================================
# 8. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if batch is None:
            continue
        
        input_values = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            if batch is None:
                continue
            
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels



# ============================================================================
# 9. TRAINING LOOP
# ============================================================================

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_loss = float('inf')
patience = 5
patience_counter = 0

print('\n' + '='*60)
print('TRAINING')
print('='*60)

for epoch in range(CONFIG['epochs']):
    print(f'\nEpoch [{epoch+1}/{CONFIG["epochs"]}]')
    
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, CONFIG['device']
    )
    
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, CONFIG['device'])
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'hubert_emotion_model.pt')
        print('  ✓ Model saved')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\n⏹ Early stopping after {epoch+1} epochs')
            break

print('\n' + '='*60)
print('✓ Training completed')



# ============================================================================
# 10. EVALUATION
# ============================================================================

model.load_state_dict(torch.load('hubert_emotion_model.pt'))

test_loss, test_acc, test_preds, test_labels = validate(
    model, test_loader, criterion, CONFIG['device']
)

print('\n' + '='*60)
print('TEST RESULTS')
print('='*60)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'\nClassification Report:\n')
print(classification_report(test_labels, test_preds, target_names=EMOTIONS))



# ============================================================================
# 11. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hubert_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ Training history plot saved')




# ============================================================================
# 12. CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Confusion Matrix - HuBERT Emotion Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('hubert_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ Confusion matrix plot saved')


# ============================================================================
# 13. PER-EMOTION METRICS
# ============================================================================

precision, recall, f1, support = precision_recall_fscore_support(
    test_labels, test_preds, labels=range(len(EMOTIONS))
)

metrics_df = pd.DataFrame({
    'Emotion': EMOTIONS,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print('\nPer-Emotion Metrics:')
print(metrics_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
metrics_df.set_index('Emotion')[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax)
plt.title('Per-Emotion Performance Metrics - HuBERT')
plt.ylabel('Score')
plt.xlabel('Emotion')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('hubert_per_emotion_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ Per-emotion metrics plot saved')



# ============================================================================
# 14. INFERENCE
# ============================================================================

def predict_emotion(audio_path, model, feature_extractor, device, sr=16000, max_length=160000):
    """Predict emotion for a single audio file."""
    model.eval()
    
    y, _ = librosa.load(audio_path, sr=sr)
    
    if len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)), mode='constant')
    else:
        y = y[:max_length]
    
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors='pt')
    input_values = inputs['input_values'].to(device)
    
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_id = torch.argmax(logits, dim=1).item()
        predicted_emotion = IDX_TO_EMOTION[predicted_id]
        confidence = probabilities[0, predicted_id].item()
    
    return {
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'probabilities': {
            EMOTIONS[i]: probabilities[0, i].item()
            for i in range(len(EMOTIONS))
        }
    }

if len(test_set) > 0:
    test_audio_path = test_set[0][0]
    true_emotion = test_set[0][1]
    
    result = predict_emotion(test_audio_path, model, feature_extractor, CONFIG['device'])
    
    print('\nExample Prediction:')
    print(f'  Audio: {os.path.basename(test_audio_path)}')
    print(f'  True Emotion: {true_emotion}')
    print(f'  Predicted Emotion: {result["predicted_emotion"]}')
    print(f'  Confidence: {result["confidence"]*100:.2f}%')
    print(f'\n  All Probabilities:')
    for emotion, prob in result['probabilities'].items():
        print(f'    {emotion}: {prob*100:.2f}%')


# ============================================================================
# 15. SAVE RESULTS
# ============================================================================

model.save_pretrained('./hubert_emotion_model')
feature_extractor.save_pretrained('./hubert_emotion_model')

results = {
    'model': 'HuBERT',
    'config': CONFIG,
    'emotions': EMOTIONS,
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'metrics': metrics_df.to_dict(),
    'confusion_matrix': cm.tolist()
}

results['config']['device'] = str(results['config']['device'])

with open('hubert_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n✓ Model and results saved successfully')
print('  - Model: ./hubert_emotion_model/')
print('  - Results: hubert_results.json')



# ============================================================================
# 16. SUMMARY
# ============================================================================

print('\n' + '='*60)
print('EMOTION RECOGNITION WITH HUBERT (SAMVEDNA) - SUMMARY')
print('='*60)
print(f'\nDataset:')
print(f'  Total samples: {len(file_paths)}')
print(f'  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')
print(f'\nModel:')
print(f'  Architecture: {CONFIG["hubert_model"]}')
print(f'  Total parameters: {total_params:,}')
print(f'  Trainable parameters: {trainable_params:,}')
print(f'\nResults:')
print(f'  Test Accuracy: {test_acc:.4f}')
print(f'  Test Loss: {test_loss:.4f}')
print(f'  Emotions: {EMOTIONS}')
print('='*60)