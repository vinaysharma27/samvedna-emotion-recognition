

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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

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
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
    'max_duration': 10.0,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'lstm_hidden_size': 256,
    'lstm_num_layers': 2,
    'dropout': 0.3,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

print('✓ Configuration loaded')
print(f'  Device: {CONFIG["device"]}')
print(f'  Emotions: {EMOTIONS}')



# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================

def extract_mfcc_features(audio_path, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512, max_duration=10.0):
    """Extract MFCC features from audio file."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta_delta])
        
        return features.T
    except Exception as e:
        print(f'Error extracting features from {audio_path}: {e}')
        return None

print('✓ Feature extraction function defined')



# ============================================================================
# 3. DATA LOADING
# ============================================================================

def load_audio_data(dataset_path, emotions=EMOTIONS):
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
                y, _ = librosa.load(str(audio_path), sr=CONFIG['sr'], duration=CONFIG['max_duration'])
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
# 4. STRATIFIED SPLIT
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
# 5. EXTRACT FEATURES
# ============================================================================

def extract_all_features(file_list, config):
    """Extract MFCC features for all audio files."""
    features_list = []
    labels_list = []
    valid_paths = []
    
    print(f'\nExtracting features from {len(file_list)} samples...')
    
    for audio_path, emotion in tqdm(file_list, desc='Feature extraction'):
        try:
            mfcc = extract_mfcc_features(
                audio_path,
                sr=config['sr'],
                n_mfcc=config['n_mfcc'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                max_duration=config['max_duration']
            )
            
            if mfcc is not None:
                features_list.append(mfcc)
                labels_list.append(EMOTION_TO_IDX[emotion])
                valid_paths.append(audio_path)
        except Exception as e:
            print(f'Error processing {audio_path}: {e}')
    
    print(f'✓ Extracted features from {len(features_list)} samples')
    return features_list, labels_list, valid_paths

print('\n' + '='*60)
print('EXTRACTING MFCC FEATURES')
print('='*60)

train_features, train_labels, train_paths = extract_all_features(train_set, CONFIG)
val_features, val_labels, val_paths = extract_all_features(val_set, CONFIG)
test_features, test_labels, test_paths = extract_all_features(test_set, CONFIG)



# ============================================================================
# 6. DATASET CLASS
# ============================================================================

class EmotionAudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return {
            'features': feature,
            'labels': label
        }

def collate_fn(batch):
    """Custom collate function to pad sequences."""
    features = [item['features'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    features_padded = pad_sequence(features, batch_first=True)
    
    return {
        'features': features_padded,
        'labels': labels
    }

train_dataset = EmotionAudioDataset(train_features, train_labels)
val_dataset = EmotionAudioDataset(val_features, val_labels)
test_dataset = EmotionAudioDataset(test_features, test_labels)

print('✓ Dataset classes created')



# ============================================================================
# 7. CREATE DATA LOADERS
# ============================================================================

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
# 8. BI-LSTM MODEL
# ============================================================================

class BiLSTMEmotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(BiLSTMEmotionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        self.bilstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        lstm_out1, _ = self.bilstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, (h_n, c_n) = self.bilstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        lstm_avg = torch.mean(lstm_out2, dim=1)
        
        dense_out = self.fc1(lstm_avg)
        dense_out = self.relu(dense_out)
        logits = self.fc2(dense_out)
        
        return logits

feature_dim = 120
model = BiLSTMEmotionModel(
    input_size=feature_dim,
    hidden_size=CONFIG['lstm_hidden_size'],
    num_layers=CONFIG['lstm_num_layers'],
    num_classes=len(EMOTIONS),
    dropout=CONFIG['dropout']
)

model = model.to(CONFIG['device'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('✓ Bi-LSTM model created')
print(f'  Total parameters: {total_params:,}')
print(f'  Trainable parameters: {trainable_params:,}')



# ============================================================================
# 9. TRAINING SETUP
# ============================================================================

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print('✓ Optimizer and scheduler configured')



# ============================================================================
# 10. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    #avg_loss = total_loss / len(dataloader)
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
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    #avg_loss = total_loss / len(dataverbose=Trueloader)
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

print('✓ Training functions defined')



# ============================================================================
# 11. TRAINING LOOP
# ============================================================================

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_loss = float('inf')
patience = 10
patience_counter = 0

print('\n' + '='*60)
print('TRAINING')
print('='*60)

for epoch in range(CONFIG['epochs']):
    print(f'\nEpoch [{epoch+1}/{CONFIG["epochs"]}]')
    
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, CONFIG['device']
    )
    
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, CONFIG['device'])
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'bilstm_emotion_model.pt')
        print('  ✓ Model saved')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\n⏹ Early stopping after {epoch+1} epochs')
            break

print('\n' + '='*60)
print('✓ Training completed')



# ============================================================================
# 12. EVALUATION
# ============================================================================

model.load_state_dict(torch.load('bilstm_emotion_model.pt'))

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
# 13. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss - Bi-LSTM')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy - Bi-LSTM')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bilstm_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ Training history plot saved')



# ============================================================================
# 14. CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Confusion Matrix - Bi-LSTM Emotion Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('bilstm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ Confusion matrix plot saved')



# ============================================================================
# 15. PER-EMOTION METRICS
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
plt.title('Per-Emotion Performance Metrics - Bi-LSTM')
plt.ylabel('Score')
plt.xlabel('Emotion')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bilstm_per_emotion_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print('✓ Per-emotion metrics plot saved')



# ============================================================================
# 16. INFERENCE
# ============================================================================

def predict_emotion(audio_path, model, config, device):
    """Predict emotion for a single audio file."""
    model.eval()
    
    mfcc = extract_mfcc_features(
        audio_path,
        sr=config['sr'],
        n_mfcc=config['n_mfcc'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        max_duration=config['max_duration']
    )
    
    if mfcc is None:
        return None
    
    features = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_id = torch.argmax(outputs, dim=1).item()
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

if len(test_paths) > 0:
    test_audio_path = test_paths[0]
    
    result = predict_emotion(test_audio_path, model, CONFIG, CONFIG['device'])
    
    if result:
        print('\nExample Prediction:')
        print(f'  Audio: {os.path.basename(test_audio_path)}')
        print(f'  Predicted Emotion: {result["predicted_emotion"]}')
        print(f'  Confidence: {result["confidence"]*100:.2f}%')
        print(f'\n  All Probabilities:')
        for emotion, prob in result['probabilities'].items():
            print(f'    {emotion}: {prob*100:.2f}%')



# ============================================================================
# 17. SAVE RESULTS
# ============================================================================

torch.save(model, 'bilstm_emotion_model_full.pth')

results = {
    'model': 'Bi-LSTM',
    'config': CONFIG,
    'emotions': EMOTIONS,
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'metrics': metrics_df.to_dict(),
    'confusion_matrix': cm.tolist()
}

results['config']['device'] = str(results['config']['device'])

with open('bilstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n✓ Model and results saved successfully')
print('  - Model: bilstm_emotion_model_full.pth')
print('  - Results: bilstm_results.json')



# ============================================================================
# 18. SUMMARY
# ============================================================================

print('\n' + '='*60)
print('EMOTION RECOGNITION WITH BI-LSTM (SAMVEDNA) - SUMMARY')
print('='*60)
print(f'\nDataset:')
print(f'  Total samples: {len(file_paths)}')
print(f'  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')
print(f'\nFeature Extraction:')
print(f'  MFCC Coefficients: {CONFIG["n_mfcc"]}')
print(f'  Total Features: 120 (MFCC + Delta + Delta-Delta)')
print(f'\nModel:')
print(f'  Architecture: Bi-LSTM')
print(f'  Hidden Size: {CONFIG["lstm_hidden_size"]}')
print(f'  Num Layers: {CONFIG["lstm_num_layers"]}')
print(f'  Total parameters: {total_params:,}')
print(f'\nResults:')
print(f'  Test Accuracy: {test_acc:.4f}')
print(f'  Test Loss: {test_loss:.4f}')
print(f'  Emotions: {EMOTIONS}')
print('='*60)