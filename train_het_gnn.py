# train_temporal_hetero_gnn.py
# =============================
# Training script for Temporal Heterogeneous GNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import numpy as np

# Import your model and data loader
from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# LOAD GRAPH AND LABELS
# ============================================================================

node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# Node labels for training (example: from anomaly assignments)
import pandas as pd
anomaly_results = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")

# Build post labels
post_labels = torch.zeros(len(node_mappings['post']), dtype=torch.long)
for _, row in anomaly_results.iterrows():
    post_id = int(row['post_id'])
    if post_id in node_mappings['post']:
        post_idx = node_mappings['post'][post_id]
        post_labels[post_idx] = 1 if row['anomaly_score'] > 0.5 else 0

# Build user labels
user_labels = torch.zeros(len(node_mappings['user']), dtype=torch.long)
user_id_to_idx = {uid: idx for uid, idx in node_mappings['user'].items()}
for user_id in user_id_to_idx.keys():
    user_posts = anomaly_results[anomaly_results['user_id'] == user_id]
    if len(user_posts) > 0:
        mean_anomaly = user_posts['anomaly_score'].mean()
        user_idx = user_id_to_idx[user_id]
        user_labels[user_idx] = 1 if mean_anomaly > 0.5 else 0

# Normalize features
for ntype, features in node_features.items():
    if torch.isnan(features).any():
        features = torch.nan_to_num(features, nan=0.0)
    if torch.isinf(features).any():
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    node_features[ntype] = F.normalize(features, p=2, dim=1).to(device)

post_labels = post_labels.to(device)
user_labels = user_labels.to(device)
edge_dict = {k: (ei.to(device), ew.to(device)) for k, (ei, ew) in edge_dict.items()}

# ============================================================================
# MODEL SETUP
# ============================================================================

node_dims = {ntype: feats.shape[1] for ntype, feats in node_features.items()}
relation_types = list(edge_dict.keys())

model = TemporalHeterogeneousGNN(
    node_dims=node_dims,
    hidden_dim=256,
    num_layers=3,
    relation_types=relation_types,
    num_classes=2
).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion_node = nn.CrossEntropyLoss()
criterion_edge = nn.BCELoss()

# Training hyperparameters
epochs = 20
edge_types_to_train = [
    ('user', 'interacts_with', 'user'),
    ('post', 'flagged_in', 'deception_cluster'),
    ('deception_cluster', 'colludes_with', 'deception_cluster')
]

best_val_loss = float('inf')
checkpoint_dir = Path("trained_models")
checkpoint_dir.mkdir(exist_ok=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(node_features, edge_dict, classify_edges=True)
    
    # NODE LOSS
    post_logits = outputs['node_logits']['post']
    user_logits = outputs['node_logits']['user']
    
    loss_post = criterion_node(post_logits, post_labels)
    loss_user = criterion_node(user_logits, user_labels)
    loss_node = loss_post + loss_user
    
    # EDGE LOSS (if labels available)
    edge_loss = 0.0
    try:
        edge_labels = torch.load("heterogeneous_graph/edge_labels.pt")
        for etype in edge_types_to_train:
            if etype in outputs['edge_scores'] and etype in edge_labels:
                scores = outputs['edge_scores'][etype]
                labels = edge_labels[etype].to(device).float()
                edge_loss += criterion_edge(scores, labels)
    except FileNotFoundError:
        print("⚠️ Edge labels not found, skipping edge loss")
    
    # TOTAL LOSS
    total_loss = loss_node + edge_loss
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    # Print progress
    print(f"Epoch {epoch}/{epochs} | Loss: {total_loss.item():.4f} | Node Loss: {loss_node.item():.4f} | Edge Loss: {edge_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss.item(),
        'config': {
            'hidden_dim': 256,
            'num_layers': 3
        }
    }, checkpoint_path)
    
    # Keep best model
    if total_loss.item() < best_val_loss:
        best_val_loss = total_loss.item()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_loss.item(),
            'config': {
                'hidden_dim': 256,
                'num_layers': 3
            }
        }, checkpoint_dir / "best_model.pt")
        print("✅ Best model updated")

print("\n✅ Training complete! Checkpoints saved in 'trained_models/'")
