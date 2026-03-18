
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# =====================================================================
# DEVICE
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================================
# LOAD DATA
# =====================================================================
print("Loading data...")

text_embeddings = torch.load("Dataset/twitter/text_aligned.pt").float()          # [N, 64]
image_embeddings = torch.load("Dataset/twitter/image_aligned.pt").float()        # [N, 64]
metadata_embeddings = torch.load("Dataset/twitter/meta_reduced.pt").float() 
labels = torch.load("Dataset/twitter/labels.pt").float()

# Flatten image/meta embeddings if necessary
if image_embeddings.ndim == 3 and image_embeddings.shape[1] == 1:
    image_embeddings = image_embeddings.squeeze(1)
if metadata_embeddings.ndim == 3 and metadata_embeddings.shape[1] == 1:
    metadata_embeddings = metadata_embeddings.squeeze(1)

print(f"Data loaded: {len(labels)} samples")
print(f"Label distribution: Real (0)={sum(labels==0).item()}, Fake (1)={sum(labels==1).item()}")

# =====================================================================
# CREATE SPLITS
# =====================================================================
indices = np.arange(len(labels))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

train_dataset = TensorDataset(
    text_embeddings[train_idx],
    image_embeddings[train_idx],
    metadata_embeddings[train_idx],
    labels[train_idx]
)
val_dataset = TensorDataset(
    text_embeddings[val_idx],
    image_embeddings[val_idx],
    metadata_embeddings[val_idx],
    labels[val_idx]
)
test_dataset = TensorDataset(
    text_embeddings[test_idx],
    image_embeddings[test_idx],
    metadata_embeddings[test_idx],
    labels[test_idx]
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =====================================================================
# LOSS FUNCTION
# =====================================================================
class SimpleFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        bce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# =====================================================================
# AUGMENTATION (NO GRAD)
# =====================================================================
class EmbeddingAugmentation:
    def __init__(self, noise_std=0.02, dropout_p=0.15):
        self.noise_std = noise_std
        self.dropout_p = dropout_p

    def __call__(self, text_emb, image_emb, meta_emb, training=True):
        if not training:
            return text_emb, image_emb, meta_emb

        with torch.no_grad():  # VERY IMPORTANT - prevent extra graph tracking
            # Gaussian noise
            text_emb = text_emb + torch.randn_like(text_emb) * self.noise_std
            image_emb = image_emb + torch.randn_like(image_emb) * self.noise_std
            meta_emb = meta_emb + torch.randn_like(meta_emb) * self.noise_std

            # Dropout
            text_mask = (torch.rand_like(text_emb) > self.dropout_p).float()
            image_mask = (torch.rand_like(image_emb) > self.dropout_p).float()
            meta_mask = (torch.rand_like(meta_emb) > self.dropout_p).float()

            text_emb = text_emb * text_mask / (1 - self.dropout_p)
            image_emb = image_emb * image_mask / (1 - self.dropout_p)
            meta_emb = meta_emb * meta_mask / (1 - self.dropout_p)

        return text_emb, image_emb, meta_emb

# =====================================================================
# MODEL
# =====================================================================
from multimodal_fakenews_model import AdaptiveMultimodalFakeNewsDetector

model = AdaptiveMultimodalFakeNewsDetector(
    d_text=text_embeddings.shape[1],
    d_image=image_embeddings.shape[1],
    d_meta=metadata_embeddings.shape[1],
    d_common=256
).to(device)

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# =====================================================================
# TRAINING CONFIG
# =====================================================================
num_epochs = 30
learning_rate = 3e-4
warmup_epochs = 3

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = SimpleFocalLoss()
augmentation = EmbeddingAugmentation()

total_steps = len(train_loader) * num_epochs
warmup_steps = len(train_loader) * warmup_epochs

def get_lr_multiplier(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

# =====================================================================
# TRAINING LOOP
# =====================================================================
history = {'train': [], 'val': []}
best_val_f1 = 0
patience_counter = 0
patience = 7

print("\nStarting training...")

for epoch in range(num_epochs):
    # ===== TRAIN =====
    model.train()
    train_loss = 0
    train_preds, train_labels_list = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        h_text, h_image, h_meta, labels_batch = [x.to(device) for x in batch]

        # Safe augmentation
        h_text, h_image, h_meta = augmentation(h_text, h_image, h_meta, training=True)

        optimizer.zero_grad()
        logits = model(h_text, h_image, h_meta, return_intermediates=False)

        loss = criterion(logits, labels_batch)
        loss.backward()  # SINGLE backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        train_labels_list.extend(labels_batch.detach().cpu().numpy())

    train_preds_binary = (np.array(train_preds) > 0.5).astype(int)
    train_labels_array = np.array(train_labels_list)
    train_f1 = f1_score(train_labels_array, train_preds_binary)
    train_acc = accuracy_score(train_labels_array, train_preds_binary)
    train_loss_avg = train_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
            h_text, h_image, h_meta, labels_batch = [x.to(device) for x in batch]
            logits = model(h_text, h_image, h_meta, return_intermediates=False)
            loss = criterion(logits, labels_batch)

            val_loss += loss.item()
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels_list.extend(labels_batch.cpu().numpy())

    val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
    val_labels_array = np.array(val_labels_list)
    val_f1 = f1_score(val_labels_array, val_preds_binary)
    val_acc = accuracy_score(val_labels_array, val_preds_binary)
    val_precision = precision_score(val_labels_array, val_preds_binary)
    val_recall = recall_score(val_labels_array, val_preds_binary)
    val_loss_avg = val_loss / len(val_loader)

    # ===== LOGGING =====
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss={train_loss_avg:.4f}, F1={train_f1:.4f}, Acc={train_acc:.4f}")
    print(f"Val   Loss={val_loss_avg:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}, P={val_precision:.4f}, R={val_recall:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

    history['train'].append({'total_loss': train_loss_avg, 'f1': train_f1, 'acc': train_acc})
    history['val'].append({'total_loss': val_loss_avg, 'f1': val_f1, 'acc': val_acc})

    # ===== CHECKPOINT =====
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model_safe.pt')
        patience_counter = 0
        print(f"  ✓ New best F1: {best_val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# =====================================================================
# TEST EVALUATION
# =====================================================================
print("\nTesting best model...")
model.load_state_dict(torch.load('best_model_safe.pt'))
model.eval()

test_preds, test_labels_list = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        h_text, h_image, h_meta, labels_batch = [x.to(device) for x in batch]
        logits = model(h_text, h_image, h_meta, return_intermediates=False)
        test_preds.extend(torch.sigmoid(logits).cpu().numpy())
        test_labels_list.extend(labels_batch.cpu().numpy())

test_preds_array = np.array(test_preds)
test_preds_binary = (test_preds_array > 0.5).astype(int)
test_labels_array = np.array(test_labels_list)

test_acc = accuracy_score(test_labels_array, test_preds_binary)
test_precision = precision_score(test_labels_array, test_preds_binary)
test_recall = recall_score(test_labels_array, test_preds_binary)
test_f1 = f1_score(test_labels_array, test_preds_binary)

try:
    test_auc = roc_auc_score(test_labels_array, test_preds_array)
except:
    test_auc = None

results = {
    'test_accuracy': float(test_acc),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'test_f1': float(test_f1),
    'test_auc': float(test_auc) if test_auc is not None else None,
    'best_val_f1': float(best_val_f1),
    'total_epochs': len(history['train']),
    'final_train_loss': history['train'][-1]['total_loss'],
    'final_val_loss': history['val'][-1]['total_loss']
}

with open('results_safe.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Training complete! Model saved to 'best_model_safe.pt'")
print("✅ Results saved to 'results_safe.json'")
