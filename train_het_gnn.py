"""
SUPERVISED TEMPORAL HETEROGENEOUS GNN TRAINING
===============================================
Trains the TemporalHeterogeneousGNN on the heterogeneous graph
using ground truth fake/real labels for post nodes.

Key design decisions:
  - Supervised node classification on post nodes (fake=1, real=0)
  - Class-weighted loss to handle imbalance (5994 fake, 4832 real)
  - Train/val/test split stratified by label
  - Early stopping on validation F1
  - Saves best model to checkpoints/best_het_gnn.pth

Outputs:
  checkpoints/best_het_gnn.pth
  gnn_results/training_curves.png
  gnn_results/gnn_post_predictions.csv
  gnn_results/gnn_evaluation_report.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

# ============================================================================
# CONFIG
# ============================================================================
HIDDEN_DIM   = 256
NUM_LAYERS   = 3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 15       # early stopping patience
BATCH_SIZE   = 512      # post nodes per gradient step
DROPOUT      = 0.3
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

output_dir = Path("gnn_results")
output_dir.mkdir(exist_ok=True)
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)

print("=" * 80)
print("SUPERVISED TEMPORAL HETEROGENEOUS GNN TRAINING")
print("=" * 80)
print(f"Device: {device}")

# ============================================================================
# STEP 1: LOAD GRAPH
# ============================================================================
print("\n[STEP 1] Loading heterogeneous graph...")

node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# Normalise and clean features
for ntype in node_features:
    feat = node_features[ntype].float()
    feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
    feat = F.normalize(feat, p=2, dim=1)
    node_features[ntype] = feat.to(device)

edge_dict_device = {
    k: (ei.to(device), ew.to(device).float())
    for k, (ei, ew) in edge_dict.items()
}

# Load post labels
post_labels = torch.load("heterogeneous_graph/post_labels.pt").to(device)
n_posts     = post_labels.shape[0]
n_fake      = post_labels.sum().item()
n_real      = n_posts - n_fake

print(f"   Posts: {n_posts} ({int(n_fake)} fake, {int(n_real)} real)")
for ntype, feat in node_features.items():
    print(f"   {ntype}: {feat.shape}")

# Load timestamps
timestamps_dict = torch.load("heterogeneous_graph/node_timestamps.pt")
timestamps_device = {k: v.float().to(device) for k, v in timestamps_dict.items()}
current_time = float(timestamps_device['post'].max().item())

# ============================================================================
# STEP 2: TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n[STEP 2] Creating train/val/test splits...")

all_indices = np.arange(n_posts)
labels_np   = post_labels.cpu().numpy()

# Stratified split: 70/15/15
train_idx, temp_idx = train_test_split(
    all_indices, test_size=0.30, stratify=labels_np, random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50,
    stratify=labels_np[temp_idx], random_state=SEED
)

train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
val_idx   = torch.tensor(val_idx,   dtype=torch.long, device=device)
test_idx  = torch.tensor(test_idx,  dtype=torch.long, device=device)

print(f"   Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
print(f"   Train fake rate: {labels_np[train_idx.cpu()].mean():.3f}")
print(f"   Val   fake rate: {labels_np[val_idx.cpu()].mean():.3f}")
print(f"   Test  fake rate: {labels_np[test_idx.cpu()].mean():.3f}")

# ============================================================================
# STEP 3: INITIALISE MODEL
# ============================================================================
print("\n[STEP 3] Initialising model...")

node_dims      = {ntype: feat.shape[1] for ntype, feat in node_features.items()}
relation_types = list(edge_dict.keys())

model = TemporalHeterogeneousGNN(
    node_dims      = node_dims,
    hidden_dim     = HIDDEN_DIM,
    num_layers     = NUM_LAYERS,
    relation_types = relation_types,
    num_classes    = 2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {total_params:,}")
print(f"   Hidden dim: {HIDDEN_DIM}, Layers: {NUM_LAYERS}")

# Class weights to handle imbalance
class_weights = torch.tensor(
    [n_fake / n_posts, n_real / n_posts],   # weight = inverse frequency
    dtype=torch.float32, device=device
)
criterion  = nn.CrossEntropyLoss(weight=class_weights)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
)

# ============================================================================
# STEP 4: FULL FORWARD PASS (graph is small enough to run all at once on MPS/CPU)
# ============================================================================

def full_forward():
    """Run one full forward pass through the GNN on the entire graph."""
    outputs = model(
        node_features  = node_features,
        edge_dict      = edge_dict_device,
        timestamps     = timestamps_device,
        current_time   = current_time,
        classify_edges = False
    )
    # Post logits: [n_posts, 2]
    post_logits = outputs['node_logits']['post']
    return post_logits


def evaluate(indices, post_logits):
    """Compute metrics on given indices."""
    logits = post_logits[indices]
    labels = post_labels[indices]

    probs = F.softmax(logits, dim=1)[:, 1]  # P(fake)
    preds = logits.argmax(dim=1)

    labels_np = labels.cpu().numpy()
    preds_np  = preds.cpu().numpy()
    probs_np  = probs.detach().cpu().numpy()

    f1  = f1_score(labels_np, preds_np, zero_division=0)
    auc = roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.5
    acc = (preds_np == labels_np).mean()
    prec = precision_score(labels_np, preds_np, zero_division=0)
    rec  = recall_score(labels_np, preds_np, zero_division=0)

    return {'f1': f1, 'auc': auc, 'acc': acc, 'precision': prec, 'recall': rec,
            'probs': probs_np, 'preds': preds_np, 'labels': labels_np}


# ============================================================================
# STEP 5: TRAINING LOOP
# ============================================================================
print("\n[STEP 4] Training...")
print(f"   {'Epoch':>6} {'Train Loss':>12} {'Val F1':>8} {'Val AUC':>8} {'Val Acc':>8} {'LR':>10}")
print(f"   {'-'*60}")

best_val_f1   = 0.0
best_epoch    = 0
patience_count = 0

train_losses, val_f1s, val_aucs = [], [], []

for epoch in range(1, EPOCHS + 1):
    model.train()

    # Mini-batch training on post nodes
    perm = torch.randperm(len(train_idx), device=device)
    shuffled_train = train_idx[perm]

    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, len(shuffled_train), BATCH_SIZE):
        batch_idx = shuffled_train[start: start + BATCH_SIZE]

        optimizer.zero_grad()
        post_logits = full_forward()

        batch_logits = post_logits[batch_idx]
        batch_labels = post_labels[batch_idx]

        loss = criterion(batch_logits, batch_labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        post_logits = full_forward()
        val_metrics = evaluate(val_idx, post_logits)

    val_f1s.append(val_metrics['f1'])
    val_aucs.append(val_metrics['auc'])

    scheduler.step(val_metrics['f1'])
    current_lr = optimizer.param_groups[0]['lr']

    if epoch % 5 == 0 or epoch == 1:
        print(f"   {epoch:>6} {avg_loss:>12.4f} {val_metrics['f1']:>8.4f} "
              f"{val_metrics['auc']:>8.4f} {val_metrics['acc']:>8.4f} {current_lr:>10.2e}")

    # Early stopping + checkpoint
    if val_metrics['f1'] > best_val_f1:
        best_val_f1  = val_metrics['f1']
        best_epoch   = epoch
        patience_count = 0
        torch.save({
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'val_f1':      val_metrics['f1'],
            'val_auc':     val_metrics['auc'],
            'node_dims':   node_dims,
            'relation_types': relation_types,
        }, ckpt_dir / "best_het_gnn.pth")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n   Early stopping at epoch {epoch} "
                  f"(best val F1={best_val_f1:.4f} at epoch {best_epoch})")
            break

print(f"\n   Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

# ============================================================================
# STEP 6: TEST EVALUATION
# ============================================================================
print("\n[STEP 5] Test evaluation (best checkpoint)...")

checkpoint = torch.load(ckpt_dir / "best_het_gnn.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

with torch.no_grad():
    post_logits = full_forward()
    test_metrics = evaluate(test_idx, post_logits)
    train_metrics = evaluate(train_idx, post_logits)

print(f"\n   Test Results:")
print(f"   Accuracy:  {test_metrics['acc']:.4f}")
print(f"   F1 (fake): {test_metrics['f1']:.4f}")
print(f"   Precision: {test_metrics['precision']:.4f}")
print(f"   Recall:    {test_metrics['recall']:.4f}")
print(f"   AUC-ROC:   {test_metrics['auc']:.4f}")

print(f"\n   Classification Report (Test):")
print(classification_report(
    test_metrics['labels'], test_metrics['preds'],
    target_names=['real', 'fake']
))

# Fake rate lift
fake_probs = test_metrics['probs']
test_labels = test_metrics['labels']

# High confidence predictions (top 25% by probability)
threshold_75 = np.percentile(fake_probs, 75)
high_conf_mask = fake_probs >= threshold_75
if high_conf_mask.sum() > 0:
    hc_fake_rate = test_labels[high_conf_mask].mean()
    baseline_rate = test_labels.mean()
    print(f"\n   High confidence predictions (top 25%):")
    print(f"   Fake rate: {hc_fake_rate:.1%} vs baseline {baseline_rate:.1%}")
    print(f"   Lift: {hc_fake_rate/baseline_rate:.2f}x")

# ============================================================================
# STEP 7: SAVE FULL PREDICTIONS
# ============================================================================
print("\n[STEP 6] Saving predictions for all posts...")

model.eval()
with torch.no_grad():
    post_logits = full_forward()
    all_probs = F.softmax(post_logits, dim=1)[:, 1].cpu().numpy()
    all_preds = post_logits.argmax(dim=1).cpu().numpy()
    all_labels = post_labels.cpu().numpy()

# Reconstruct post_ids from node mappings
post_id_from_idx = {v: k for k, v in node_mappings['post'].items()}
post_ids_ordered = [post_id_from_idx.get(i, f"unknown_{i}") for i in range(n_posts)]

# Determine split membership
split_membership = np.full(n_posts, 'train', dtype=object)
split_membership[val_idx.cpu().numpy()]  = 'val'
split_membership[test_idx.cpu().numpy()] = 'test'

predictions_df = pd.DataFrame({
    'post_id':      post_ids_ordered,
    'true_label':   ['fake' if l == 1 else 'real' for l in all_labels],
    'predicted':    ['fake' if p == 1 else 'real' for p in all_preds],
    'fake_prob':    all_probs,
    'correct':      all_labels == all_preds,
    'split':        split_membership,
})

predictions_df.to_csv(output_dir / "gnn_post_predictions.csv", index=False)
print(f"   ✅ Saved gnn_post_predictions.csv ({len(predictions_df)} posts)")

# ============================================================================
# STEP 8: TRAINING CURVES
# ============================================================================
print("\n[STEP 7] Generating training curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Temporal Heterogeneous GNN — Training', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(train_losses, color='steelblue', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(val_f1s, color='coral', linewidth=2, label='Val F1')
ax.axvline(best_epoch - 1, color='green', linestyle='--', label=f'Best (ep {best_epoch})')
ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score')
ax.set_title('Validation F1'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(val_aucs, color='mediumpurple', linewidth=2)
ax.axhline(0.5, color='gray', linestyle='--', label='Random')
ax.set_xlabel('Epoch'); ax.set_ylabel('AUC-ROC')
ax.set_title('Validation AUC-ROC'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved training_curves.png")

# ============================================================================
# STEP 9: REPORT
# ============================================================================
report = f"""
TEMPORAL HETEROGENEOUS GNN — EVALUATION REPORT
{'='*60}

Model Configuration:
  Hidden dim:    {HIDDEN_DIM}
  Layers:        {NUM_LAYERS}
  Parameters:    {total_params:,}
  Training epochs: {best_epoch} (best)

Graph Structure:
  Node types:  {len(node_features)}
  Edge types:  {len(edge_dict)}
  Post nodes:  {n_posts} ({int(n_fake)} fake, {int(n_real)} real)

Train/Val/Test Split (70/15/15):
  Train: {len(train_idx)}
  Val:   {len(val_idx)}
  Test:  {len(test_idx)}

Test Performance:
  Accuracy:  {test_metrics['acc']:.4f}
  F1 (fake): {test_metrics['f1']:.4f}
  Precision: {test_metrics['precision']:.4f}
  Recall:    {test_metrics['recall']:.4f}
  AUC-ROC:   {test_metrics['auc']:.4f}

Classification Report (Test):
{classification_report(test_metrics['labels'], test_metrics['preds'],
                        target_names=['real', 'fake'])}
"""

with open(output_dir / "gnn_evaluation_report.txt", 'w') as f:
    f.write(report)
print(f"   ✅ Saved gnn_evaluation_report.txt")

print("\n" + "=" * 80)
print("✅ GNN TRAINING COMPLETE")
print("=" * 80)
print(f"""
Summary:
  Best val F1:  {best_val_f1:.4f} (epoch {best_epoch})
  Test F1:      {test_metrics['f1']:.4f}
  Test AUC:     {test_metrics['auc']:.4f}
  Test Acc:     {test_metrics['acc']:.4f}

Saved:
  checkpoints/best_het_gnn.pth
  gnn_results/gnn_post_predictions.csv
  gnn_results/training_curves.png
  gnn_results/gnn_evaluation_report.txt

Next steps:
  python3 analyze_dected_posts.py
  python3 campaign_investigator.py
  python3 explainability.py
""")