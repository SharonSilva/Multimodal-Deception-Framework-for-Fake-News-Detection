"""
UNSUPERVISED SUSPICIOUS CONTENT DETECTION
==========================================
Instead of supervised training with noisy labels, use:
1. Train GNN to learn good embeddings (unsupervised)
2. Use embeddings + graph features to detect suspicious content
3. Multiple detection strategies: clustering, isolation forest, graph metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.stats import zscore

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# LOAD GRAPH & TRAIN UNSUPERVISED
# ============================================================================
print("="*80)
print("UNSUPERVISED SUSPICIOUS CONTENT DETECTION")
print("="*80)

print("\n📦 Loading heterogeneous graph...")
node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# Normalize features
for ntype, features in node_features.items():
    features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    node_features[ntype] = F.normalize(features, p=2, dim=1).to(device)

edge_dict = {k: (ei.to(device), ew.to(device)) for k, (ei, ew) in edge_dict.items()}

# Load ground truth for evaluation only
anomaly_results = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")
post_true_labels = {}
for _, row in anomaly_results.iterrows():
    post_id = int(row['post_id'])
    if post_id in node_mappings['post']:
        post_true_labels[post_id] = row['anomaly_score']

print(f"✅ Loaded {len(node_features)} node types")

# ============================================================================
# TRAIN GNN TO LEARN EMBEDDINGS (RECONSTRUCTION OBJECTIVE)
# ============================================================================
print("\n🏗️ Training GNN with reconstruction objective...")

node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}
relation_types = list(edge_dict.keys())

model = TemporalHeterogeneousGNN(
    node_dims=node_dims,
    hidden_dim=256,
    num_layers=3,
    relation_types=relation_types,
    num_classes=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Reconstruction loss: predict edge existence
def reconstruction_loss(embeddings, edge_dict):
    total_loss = 0
    num_edges = 0
    
    for (src_type, rel, dst_type), (edge_index, edge_weight) in edge_dict.items():
        if edge_index.shape[1] == 0:
            continue
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Validate indices
        if src_type not in embeddings or dst_type not in embeddings:
            continue
        
        src_max = embeddings[src_type].shape[0]
        dst_max = embeddings[dst_type].shape[0]
        
        valid = (src_nodes < src_max) & (dst_nodes < dst_max)
        if valid.sum() == 0:
            continue
        
        src_nodes = src_nodes[valid]
        dst_nodes = dst_nodes[valid]
        
        # Get embeddings
        src_emb = embeddings[src_type][src_nodes]
        dst_emb = embeddings[dst_type][dst_nodes]
        
        # Predict edge existence (dot product + sigmoid)
        pred = torch.sigmoid((src_emb * dst_emb).sum(dim=1))
        target = torch.ones_like(pred)
        
        loss = F.binary_cross_entropy(pred, target)
        total_loss += loss
        num_edges += 1
    
    return total_loss / max(num_edges, 1)

print("Training for 30 epochs...")
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(node_features, edge_dict, classify_edges=False)
    loss = reconstruction_loss(outputs['embeddings'], edge_dict)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30, Loss: {loss.item():.4f}")

print("✅ Training complete!\n")

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================
print("🎯 Extracting learned embeddings...")

model.eval()
with torch.no_grad():
    outputs = model(node_features, edge_dict, classify_edges=False)
    embeddings = outputs['embeddings']

post_embeddings = embeddings['post'].cpu().numpy()
user_embeddings = embeddings['user'].cpu().numpy()

print(f"✅ Extracted embeddings:")
print(f"   Posts: {post_embeddings.shape}")
print(f"   Users: {user_embeddings.shape}")

# ============================================================================
# COMPUTE GRAPH-BASED SUSPICION SCORES
# ============================================================================
print("\n📊 Computing graph-based suspicion scores...")

# Method 1: Isolation Forest (anomaly detection)
print("\n[1] Isolation Forest Anomaly Detection")
iso_forest = IsolationForest(contamination=0.15, random_state=42)
post_anomaly_scores = -iso_forest.fit_predict(post_embeddings)  # -1 or 1, convert to 0/1
post_anomaly_scores = (post_anomaly_scores + 1) / 2  # Convert to 0/1

suspicious_by_iso = (post_anomaly_scores == 1).sum()
print(f"   Flagged {suspicious_by_iso}/{len(post_anomaly_scores)} posts as suspicious")

# Method 2: DBSCAN Clustering (outliers)
print("\n[2] DBSCAN Clustering (outliers = suspicious)")
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(post_embeddings)

outliers = (clusters == -1)
suspicious_by_dbscan = outliers.sum()
print(f"   Found {suspicious_by_dbscan} outlier posts")

# Method 3: Local Outlier Factor
print("\n[3] Computing embedding distance to neighbors")
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=20, metric='cosine').fit(post_embeddings)
distances, indices = nbrs.kneighbors(post_embeddings)

# Mean distance to k-nearest neighbors (high = suspicious)
mean_distances = distances.mean(axis=1)
distance_threshold = np.percentile(mean_distances, 85)  # Top 15%
suspicious_by_distance = mean_distances > distance_threshold

print(f"   Flagged {suspicious_by_distance.sum()} posts with high neighbor distance")

# Method 4: Graph centrality (using edge connections)
print("\n[4] Graph-based centrality scores")

# Get post-to-post connections from deception clusters
post_deception_edges = edge_dict.get(('post', 'flagged_in', 'deception_cluster'), None)

if post_deception_edges is not None:
    edge_index, _ = post_deception_edges
    posts_in_deception = set(edge_index[0].cpu().numpy())
    suspicious_by_deception = np.array([i in posts_in_deception for i in range(len(post_embeddings))])
    print(f"   {suspicious_by_deception.sum()} posts in deception clusters")
else:
    suspicious_by_deception = np.zeros(len(post_embeddings), dtype=bool)

# ============================================================================
# ENSEMBLE SUSPICION SCORE
# ============================================================================
print("\n🎲 Creating ensemble suspicion score...")

# Combine all methods (voting)
suspicion_votes = (
    post_anomaly_scores.astype(int) +
    outliers.astype(int) +
    suspicious_by_distance.astype(int) +
    suspicious_by_deception.astype(int)
)

# Normalize to 0-1
suspicion_scores = suspicion_votes / 4.0

print(f"\n📊 Suspicion Score Distribution:")
print(f"   Mean: {suspicion_scores.mean():.4f}")
print(f"   Std:  {suspicion_scores.std():.4f}")
print(f"   Min:  {suspicion_scores.min():.4f}")
print(f"   Max:  {suspicion_scores.max():.4f}")

# Threshold at top 15%
suspicion_threshold = np.percentile(suspicion_scores, 85)
final_suspicious = suspicion_scores >= suspicion_threshold

print(f"\n✅ Final detection:")
print(f"   Threshold: {suspicion_threshold:.4f}")
print(f"   Flagged: {final_suspicious.sum()}/{len(final_suspicious)} posts as suspicious")

# ============================================================================
# EVALUATE AGAINST GROUND TRUTH
# ============================================================================
print("\n🎯 Evaluating against ground truth...")

# Map back to original post IDs
post_id_to_idx = {pid: idx for pid, idx in node_mappings['post'].items()}
idx_to_post_id = {idx: pid for pid, idx in post_id_to_idx.items()}

detected_post_ids = [idx_to_post_id[i] for i in range(len(final_suspicious)) if final_suspicious[i]]

# Compare with ground truth
true_positives = 0
false_positives = 0
total_detected = 0

for post_id in detected_post_ids:
    total_detected += 1
    if post_id in post_true_labels:
        if post_true_labels[post_id] > 0.5:  # High anomaly score
            true_positives += 1
        else:
            false_positives += 1

# Count true suspicious posts in dataset
true_suspicious_total = sum(1 for score in post_true_labels.values() if score > 0.5)

precision = true_positives / (total_detected + 1e-10)
recall = true_positives / (true_suspicious_total + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)

print(f"\n📊 Detection Performance:")
print(f"   Detected: {total_detected} posts")
print(f"   True Positives: {true_positives}")
print(f"   False Positives: {false_positives}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n💾 Saving results...")

results_dir = Path("suspicious_detection_results")
results_dir.mkdir(exist_ok=True)

# Create results dataframe
results = []
for i in range(len(suspicion_scores)):
    post_id = idx_to_post_id.get(i)
    if post_id is None:
        continue
    
    results.append({
        'post_id': post_id,
        'suspicion_score': suspicion_scores[i],
        'is_suspicious': final_suspicious[i],
        'iso_forest_flag': post_anomaly_scores[i],
        'dbscan_outlier': outliers[i],
        'high_distance': suspicious_by_distance[i],
        'in_deception_cluster': suspicious_by_deception[i],
        'true_anomaly_score': post_true_labels.get(post_id, np.nan)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(results_dir / "suspicious_posts_detected.csv", index=False)

print(f"✅ Saved results to {results_dir / 'suspicious_posts_detected.csv'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Suspicion score distribution
ax = axes[0, 0]
ax.hist(suspicion_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(suspicion_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {suspicion_threshold:.2f}')
ax.set_xlabel('Suspicion Score')
ax.set_ylabel('Frequency')
ax.set_title('Suspicion Score Distribution')
ax.legend()
ax.set_yscale('log')

# 2. Detection method comparison
ax = axes[0, 1]
methods = ['Iso Forest', 'DBSCAN', 'Distance', 'Deception\nCluster', 'Final']
counts = [
    post_anomaly_scores.sum(),
    outliers.sum(),
    suspicious_by_distance.sum(),
    suspicious_by_deception.sum(),
    final_suspicious.sum()
]
ax.bar(methods, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'], edgecolor='black')
ax.set_ylabel('Number of Suspicious Posts')
ax.set_title('Detection Method Comparison')
ax.grid(axis='y', alpha=0.3)

# 3. Score vs ground truth
ax = axes[1, 0]
true_scores = []
pred_scores = []
for i in range(len(suspicion_scores)):
    post_id = idx_to_post_id.get(i)
    if post_id in post_true_labels:
        true_scores.append(post_true_labels[post_id])
        pred_scores.append(suspicion_scores[i])

ax.scatter(true_scores, pred_scores, alpha=0.5, s=20)
ax.set_xlabel('True Anomaly Score')
ax.set_ylabel('Predicted Suspicion Score')
ax.set_title('Predicted vs True Scores')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Top suspicious posts
ax = axes[1, 1]
top_n = 20
top_indices = np.argsort(suspicion_scores)[-top_n:][::-1]
top_scores = suspicion_scores[top_indices]

ax.barh(range(top_n), top_scores, color='crimson', edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"Post {idx_to_post_id.get(i, '?')}" for i in top_indices], fontsize=8)
ax.set_xlabel('Suspicion Score')
ax.set_title(f'Top {top_n} Most Suspicious Posts')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(results_dir / 'detection_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization to {results_dir / 'detection_analysis.png'}")

print("\n" + "="*80)
print("✅ UNSUPERVISED DETECTION COMPLETE!")
print("="*80)
print(f"\n📁 Results saved to: {results_dir}/")
print(f"   • suspicious_posts_detected.csv - Detection results")
print(f"   • detection_analysis.png - Visualizations")
print(f"\n🎯 Summary:")
print(f"   Detected {final_suspicious.sum()} suspicious posts")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")