

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Import your fusion layer
from rough_work import EmotionAwareFusionLayer, EmotionAwareFakeNewsDetector

# ============================================================================
# DEVICE SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n Using device: {device.type.upper()}")

if device.type == "cuda":
    print(f"  -> GPU: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print("  -> MPS (Metal Performance Shaders) active for Mac acceleration.")
else:
    print("  -> Running on CPU")

# ============================================================================
# DATASET CLASS WITH DIMENSION FIXING
# ============================================================================
class FakeNewsVADDataset(Dataset):
    """Dataset class for emotion-aware fake news detection with dimension fixes."""

    def __init__(self, df, vad_data, text_features, image_features, metadata_features):
        self.df = df
        self.vad_text = vad_data['vad_text']
        self.vad_image = vad_data['vad_image']
        self.affective_meta = vad_data['affective_meta']

        # FIX: Ensure all features are 2D tensors
        self.text_features = self._ensure_2d(text_features)
        self.image_features = self._ensure_2d(image_features)
        self.metadata_features = self._ensure_2d(metadata_features)

        # Convert labels to numeric
        if self.df['label'].dtype == object:
            print("  Converting object labels to numeric (0/1)...")
            self.df['label'] = self.df['label'].map({'fake': 1, 'real': 0})
        
        self.df['label'] = pd.to_numeric(self.df['label'], errors='coerce').fillna(0)
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float32)

    def _ensure_2d(self, tensor):
        """Ensure tensor is 2D by squeezing middle dimensions."""
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        elif tensor.ndim == 1:
            return tensor.unsqueeze(1)
        return tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'text_features': self.text_features[idx],
            'image_features': self.image_features[idx],
            'metadata_features': self.metadata_features[idx],
            'vad_text': self.vad_text[idx],
            'vad_image': self.vad_image[idx],
            'affective_meta': self.affective_meta[idx],
            'labels': self.labels[idx]
        }

# ============================================================================
# EMOTION-AWARE MODEL WITH DIMENSION SAFETY
# ============================================================================
#removed

# ============================================================================
# SAFE VAD DATA INDEXING
# ============================================================================
def index_vad_data(vad_data, indices):
    """Safely index VAD data with dimension validation."""
    indexed_data = {}
    
    if torch.is_tensor(indices):
        indices = indices.tolist()
    elif isinstance(indices, np.ndarray):
        indices = indices.tolist()
    
    print(f"\n Indexing VAD data for {len(indices)} samples...")
    
    for key, value in vad_data.items():
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=torch.float32)
        elif not torch.is_tensor(value):
            raise ValueError(f"Unsupported type for vad_data['{key}']: {type(value)}")
        
        print(f"  {key}: shape {value.shape}")
        
        if value.shape[0] != len(vad_data['vad_text']):
            raise ValueError(
                f"Dimension mismatch for '{key}': "
                f"Expected {len(vad_data['vad_text'])} samples, got {value.shape[0]}"
            )
        
        indexed_data[key] = value[indices]
        print(f"    → Indexed: {indexed_data[key].shape}")
    
    return indexed_data

# ============================================================================
# VAD DATA PREPARATION
# ============================================================================
def prepare_vad_data_with_validation(df):
    """Prepare VAD data with automatic dimension adjustment."""
    N = len(df)
    print(f"\nPreparing VAD data for {N} samples...")

    # Text VAD
    if all(col in df.columns for col in ['text_valence', 'text_arousal', 'text_dominance']):
        vad_text = torch.tensor(
            df[['text_valence', 'text_arousal', 'text_dominance']].values,
            dtype=torch.float32
        )
        print(f"   Text VAD: {vad_text.shape}")
    else:
        print("   Text VAD columns not found, using random values")
        vad_text = torch.rand(N, 3)

    # Image VAD
    vad_image = None
    if os.path.exists("Dataset/twitter/image_vad.pt"):
        vad_image = torch.load("Dataset/twitter/image_vad.pt").float()
        print(f"  Image VAD loaded: {vad_image.shape}")
    elif os.path.exists("Dataset/twitter/scene_emotions_vad_proj.csv"):
        df_scene = pd.read_csv("Dataset/twitter/scene_emotions_vad_proj.csv")
        if all(col in df_scene.columns for col in ['valence', 'arousal', 'dominance']):
            vad_image = torch.tensor(
                df_scene[['valence', 'arousal', 'dominance']].values,
                dtype=torch.float32
            )
            print(f"  Scene VAD: {vad_image.shape}")
    
    if vad_image is None:
        print("    Image VAD not found, using random values")
        vad_image = torch.rand(N, 3)

    # Adjust image VAD size
    if vad_image.shape[0] != N:
        print(f"    Image VAD size mismatch: {vad_image.shape[0]} vs {N}, adjusting...")
        if vad_image.shape[0] > N:
            vad_image = vad_image[:N]
        else:
            repeat_factor = int(np.ceil(N / vad_image.shape[0]))
            vad_image = vad_image.repeat(repeat_factor, 1)[:N]
    print(f"   Image VAD adjusted: {vad_image.shape}")

    # Affective metadata
    if os.path.exists("Dataset/affectnet/affective_embedding.npy"):
        affective_meta = torch.tensor(np.load("Dataset/affectnet/affective_embedding.npy"), dtype=torch.float32)
        if affective_meta.shape[0] != N:
            print(f"    Affective embeddings mismatch: {affective_meta.shape[0]} vs {N}, adjusting...")
            if affective_meta.shape[0] > N:
                affective_meta = affective_meta[:N]
            else:
                repeat_factor = int(np.ceil(N / affective_meta.shape[0]))
                affective_meta = affective_meta.repeat(repeat_factor, 1)[:N]
        print(f"   Affective meta adjusted: {affective_meta.shape}")
    else:
        print("    Affective embeddings not found, using random values")
        affective_meta = torch.randn(N, 128)

    vad_data = {
        'vad_text': vad_text,
        'vad_image': vad_image,
        'affective_meta': affective_meta
    }

    print(f"\n VAD data prepared successfully for {N} samples")
    return vad_data

# ============================================================================
# TRAINING & VALIDATION
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    progress = tqdm(dataloader, desc=" Training", leave=False)

    for batch in progress:
        h_text = batch['text_features'].to(device)
        h_image = batch['image_features'].to(device)
        h_meta = batch['metadata_features'].to(device)
        vad_text = batch['vad_text'].to(device)
        vad_image = batch['vad_image'].to(device)
        affective_meta = batch['affective_meta'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits, intermediates = model(h_text, h_image, h_meta, affective_meta, vad_text, vad_image)
        loss_main = criterion(logits.squeeze(-1), labels)

        # Auxiliary losses
        v_mismatch = intermediates['v_mismatch']
        congruence = intermediates['congruence'].squeeze()
        
        # Ensure v_mismatch is 2D
        if v_mismatch.ndim > 2:
            v_mismatch = v_mismatch.squeeze(1)
        
        mismatch_magnitude = torch.norm(v_mismatch, dim=1)
        target_magnitude = labels * 2.0
        loss_mismatch = nn.MSELoss()(mismatch_magnitude, target_magnitude)
        loss_congruence = nn.MSELoss()(1.0 - congruence, labels)

        # Total loss
        loss = loss_main + 0.1 * loss_mismatch + 0.05 * loss_congruence
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        progress = tqdm(dataloader, desc="🔍 Validating", leave=False)
        for batch in progress:
            h_text = batch['text_features'].to(device)
            h_image = batch['image_features'].to(device)
            h_meta = batch['metadata_features'].to(device)
            vad_text = batch['vad_text'].to(device)
            vad_image = batch['vad_image'].to(device)
            affective_meta = batch['affective_meta'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(h_text, h_image, h_meta, affective_meta, vad_text, vad_image)
            loss = criterion(logits.squeeze(-1), labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader), correct / total

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" TRAINING EMOTION-AWARE FAKE NEWS DETECTOR")
    print("="*70)

    # Load dataset
    print("\n Loading dataset...")
    df = pd.read_pickle("Dataset/twitter/df_with_text_emotions_vad.pkl")
    df = df.iloc[:11844].reset_index(drop=True)
    print(f"  Loaded {len(df)} samples")
    
    # Prepare VAD data
    vad_data = prepare_vad_data_with_validation(df)
    torch.save(vad_data, "Dataset/twitter/prepared_vad_data_validated.pt")
    print("\n Saved validated VAD data")

    # Load features with dimension fixing
    print("\n Loading feature embeddings...")
    
    # Text features
    if 'semantic_vector' in df.columns:
        text_features = torch.tensor(np.stack(df['semantic_vector'].values), dtype=torch.float32)
    else:
        print("    Using random text features")
        text_features = torch.randn(len(df), 128)
    print(f"  Text features: {text_features.shape}")
    
    # Image features
    if os.path.exists("Dataset/twitter/image_embeddings_cache.pkl"):
        import pickle
        with open("Dataset/twitter/image_embeddings_cache.pkl", "rb") as f:
            cache = pickle.load(f)
        image_features = cache['image_embeddings']
        if not torch.is_tensor(image_features):
            image_features = torch.tensor(image_features, dtype=torch.float32)
    else:
        print("    Using random image features")
        image_features = torch.randn(len(df), 1024)
    print(f"   Image features: {image_features.shape}")
    
    # Metadata features - FIX DIMENSIONS
    if os.path.exists("metadata_user_sequence_embeddings.pt"):
        metadata_features = torch.load("metadata_user_sequence_embeddings.pt")
        # FIX: Squeeze middle dimension if (N, 1, 128)
        if metadata_features.ndim == 3 and metadata_features.shape[1] == 1:
            metadata_features = metadata_features.squeeze(1)
            print(f"  Squeezed metadata from 3D to 2D")
    else:
        print("    Using random metadata features")
        metadata_features = torch.randn(len(df), 128)
    print(f"   Metadata features: {metadata_features.shape}")

    # Split train/val
    print("\n  Splitting train/validation...")
    train_idx, val_idx = train_test_split(
        range(len(df)), 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    train_vad_data = index_vad_data(vad_data, train_idx)
    val_vad_data = index_vad_data(vad_data, val_idx)

    train_dataset = FakeNewsVADDataset(
        df.iloc[train_idx].reset_index(drop=True),
        train_vad_data,
        text_features[train_idx],
        image_features[train_idx],
        metadata_features[train_idx]
    )
    
    val_dataset = FakeNewsVADDataset(
        df.iloc[val_idx].reset_index(drop=True),
        val_vad_data,
        text_features[val_idx],
        image_features[val_idx],
        metadata_features[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Model
    print("\n Initializing model...")
    model = EmotionAwareFakeNewsDetector().to(device)
    print(f" Model initialized ({sum(p.numel() for p in model.parameters()):,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    num_epochs = 20
    print("\n Starting training...\n")

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"   Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_emotion_aware_detector.pth")
            print(f"  Saved new best model (Val Acc: {val_acc:.4f})")

    print("\n" + "="*70)
    print(f" TRAINING COMPLETE — Best Val Acc: {best_val_acc:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()