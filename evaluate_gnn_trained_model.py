"""
evaluate_trained_model.py
==========================
Comprehensive evaluation of the trained Temporal Heterogeneous GNN model.

Includes:
- Node classification metrics with adjusted thresholds
- Edge classification analysis
- Community risk assessment
- Deception cluster detection
- Visualization of results
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)

# Import model
from temporal_graph import (
    TemporalHeterogeneousGNN,
    load_heterogeneous_graph
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

print("="*80)
print("LOADING MODEL AND DATA")
print("="*80)

# Load graph
node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# Normalize features (same as training)
for ntype, features in node_features.items():
    if torch.isnan(features).any():
        features = torch.nan_to_num(features, nan=0.0)
    if torch.isinf(features).any():
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    node_features[ntype] = F.normalize(features, p=2, dim=1)

# Move to device
node_features = {k: v.to(device) for k, v in node_features.items()}
edge_dict = {k: (ei.to(device), ew.to(device)) for k, (ei, ew) in edge_dict.items()}

# Load model
checkpoint = torch.load('trained_models/best_model.pt', map_location=device)

node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}
relation_types = list(edge_dict.keys())

model = TemporalHeterogeneousGNN(
    node_dims=node_dims,
    hidden_dim=checkpoint['config']['hidden_dim'],
    num_layers=checkpoint['config']['num_layers'],
    relation_types=relation_types,
    num_classes=2
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
print(f"   Training loss: {checkpoint.get('train_loss', 'N/A')}")
print(f"   Validation loss: {checkpoint['val_loss']:.4f}")

# Load labels
anomaly_results = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")

post_labels = torch.zeros(len(node_mappings['post']), dtype=torch.long)
for _, row in anomaly_results.iterrows():
    post_id = int(row['post_id'])
    if post_id in node_mappings['post']:
        post_idx = node_mappings['post'][post_id]
        post_labels[post_idx] = 1 if row['anomaly_score'] > 0.5 else 0

user_labels = torch.zeros(len(node_mappings['user']), dtype=torch.long)
user_id_to_idx = {uid: idx for uid, idx in node_mappings['user'].items()}

for user_id in user_id_to_idx.keys():
    user_posts = anomaly_results[anomaly_results['user_id'] == user_id]
    if len(user_posts) > 0:
        mean_anomaly = user_posts['anomaly_score'].mean()
        user_idx = user_id_to_idx[user_id]
        user_labels[user_idx] = 1 if mean_anomaly > 0.5 else 0

post_labels = post_labels.to(device)
user_labels = user_labels.to(device)

# ============================================================================
# RUN INFERENCE
# ============================================================================

print("\n" + "="*80)
print("RUNNING INFERENCE")
print("="*80)

with torch.no_grad():
    outputs = model(node_features, edge_dict, classify_edges=True)

print("✅ Inference complete")

# ============================================================================
# NODE CLASSIFICATION EVALUATION (WITH PROBABILITY ADJUSTMENT)
# ============================================================================

print("\n" + "="*80)
print("NODE CLASSIFICATION EVALUATION")
print("="*80)

# Post classification with probability scores
post_logits = outputs['node_logits']['post']
post_probs = F.softmax(post_logits, dim=1)
post_scores = post_probs[:, 1]  # Probability of being suspicious

# User classification
user_logits = outputs['node_logits']['user']
user_probs = F.softmax(user_logits, dim=1)
user_scores = user_probs[:, 1]

# Evaluate at different thresholds
print("\n📊 Post Classification (at different thresholds):")
print("-" * 60)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    post_pred = (post_scores > thresh).long()
    
    tp = ((post_pred == 1) & (post_labels == 1)).sum().item()
    fp = ((post_pred == 1) & (post_labels == 0)).sum().item()
    tn = ((post_pred == 0) & (post_labels == 0)).sum().item()
    fn = ((post_pred == 0) & (post_labels == 1)).sum().item()
    
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    
    print(f"Threshold {thresh:.1f}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(f"  Predicted suspicious: {(post_pred == 1).sum().item()}/{len(post_pred)}")

print("\n📊 User Classification (at different thresholds):")
print("-" * 60)

for thresh in thresholds:
    user_pred = (user_scores > thresh).long()
    
    tp = ((user_pred == 1) & (user_labels == 1)).sum().item()
    fp = ((user_pred == 1) & (user_labels == 0)).sum().item()
    tn = ((user_pred == 0) & (user_labels == 0)).sum().item()
    fn = ((user_pred == 0) & (user_labels == 1)).sum().item()
    
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    
    print(f"Threshold {thresh:.1f}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(f"  Predicted suspicious: {(user_pred == 1).sum().item()}/{len(user_pred)}")

# ============================================================================
# EDGE CLASSIFICATION EVALUATION
# ============================================================================

print("\n" + "="*80)
print("EDGE CLASSIFICATION EVALUATION")
print("="*80)

edge_labels = torch.load("heterogeneous_graph/edge_labels.pt")

for etype, scores in outputs['edge_scores'].items():
    if etype in edge_labels:
        labels = edge_labels[etype].to(device)
        
        print(f"\n📊 {etype}:")
        print("-" * 60)
        
        # Evaluate at different thresholds
        for thresh in [0.4, 0.5, 0.6]:
            pred = (scores > thresh).long()
            
            tp = ((pred == 1) & (labels == 1)).sum().item()
            fp = ((pred == 1) & (labels == 0)).sum().item()
            tn = ((pred == 0) & (labels == 0)).sum().item()
            fn = ((pred == 0) & (labels == 1)).sum().item()
            
            acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1 = 2 * prec * rec / (prec + rec + 1e-10)
            
            print(f"Threshold {thresh:.1f}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# ============================================================================
# COMMUNITY RISK SCORES
# ============================================================================

print("\n" + "="*80)
print("COMMUNITY RISK ASSESSMENT")
print("="*80)

if 'community_risk' in outputs:
    community_risks = outputs['community_risk'].cpu().numpy()
    
    print("\n📊 Community Risk Scores:")
    for i, risk in enumerate(community_risks):
        print(f"   Community {i}: Risk = {risk:.4f}")

# ============================================================================
# DECEPTION CLUSTER SCORES
# ============================================================================

print("\n" + "="*80)
print("DECEPTION CLUSTER DETECTION")
print("="*80)

if 'deception_score' in outputs:
    deception_scores = outputs['deception_score'].cpu().numpy()
    
    print(f"\n📊 Deception Cluster Statistics:")
    print(f"   Mean score: {deception_scores.mean():.4f}")
    print(f"   Std dev: {deception_scores.std():.4f}")
    print(f"   Min score: {deception_scores.min():.4f}")
    print(f"   Max score: {deception_scores.max():.4f}")
    
    # Top suspicious clusters
    top_k = min(10, len(deception_scores))
    top_indices = np.argsort(deception_scores)[-top_k:][::-1]
    
    print(f"\n   Top {top_k} Most Suspicious Clusters:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"      {rank}. Cluster {idx}: Score = {deception_scores[idx]:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

output_dir = Path("evaluation_results")
output_dir.mkdir(exist_ok=True)

# 1. Score distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Post scores
ax = axes[0, 0]
post_scores_np = post_scores.cpu().numpy()
ax.hist(post_scores_np[post_labels.cpu() == 0], bins=50, alpha=0.7, label='Normal', color='blue')
ax.hist(post_scores_np[post_labels.cpu() == 1], bins=50, alpha=0.7, label='Suspicious', color='red')
ax.set_xlabel('Suspicion Score')
ax.set_ylabel('Frequency')
ax.set_title('Post Suspicion Score Distribution')
ax.legend()
ax.set_yscale('log')

# User scores
ax = axes[0, 1]
user_scores_np = user_scores.cpu().numpy()
ax.hist(user_scores_np[user_labels.cpu() == 0], bins=50, alpha=0.7, label='Normal', color='blue')
ax.hist(user_scores_np[user_labels.cpu() == 1], bins=50, alpha=0.7, label='Suspicious', color='red')
ax.set_xlabel('Suspicion Score')
ax.set_ylabel('Frequency')
ax.set_title('User Suspicion Score Distribution')
ax.legend()
ax.set_yscale('log')

# Community risks
ax = axes[1, 0]
if 'community_risk' in outputs:
    ax.bar(range(len(community_risks)), community_risks, color='orange', edgecolor='black')
    ax.set_xlabel('Community ID')
    ax.set_ylabel('Risk Score')
    ax.set_title('Community Risk Scores')
    ax.set_ylim([0, 1])

# Deception cluster scores
ax = axes[1, 1]
if 'deception_score' in outputs:
    ax.hist(deception_scores, bins=30, color='purple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Deception Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Deception Cluster Score Distribution')

plt.tight_layout()
plt.savefig(output_dir / 'score_distributions.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved score distributions")

# 2. ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Post ROC
ax = axes[0]
fpr, tpr, _ = roc_curve(post_labels.cpu().numpy(), post_scores_np)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Post Classification ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# User ROC
ax = axes[1]
fpr, tpr, _ = roc_curve(user_labels.cpu().numpy(), user_scores_np)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('User Classification ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved ROC curves")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print(f"\n📁 Results saved to: {output_dir}/")
print("   • score_distributions.png - Score histograms")
print("   • roc_curves.png - ROC curves for classification")