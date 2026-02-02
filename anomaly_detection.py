""""
TEMPORAL-AWARE CAMPAIGN DETECTION
==================================
Detects coordinated disinformation campaigns by finding groups of posts that are:
- Content-similar (high embedding similarity)
- Time-clustered (posted close together)
- Anomalous (weighted by suspicion scores)

Uses:
- Post embeddings (z_out)
- Timestamps
- Anomaly scores
- Community detection (Louvain algorithm)
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import networkx as nx
try:
    import community as community_louvain
except ImportError:
    print("⚠️  Installing python-louvain...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-louvain'])
    import community as community_louvain

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("TEMPORAL-AWARE CAMPAIGN DETECTION PIPELINE")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'similarity_threshold': 0.5,      # Min cosine similarity to create edge
    'time_decay_tau': 3600,           # Time decay in seconds (1 hour)
    'min_campaign_size': 3,           # Min posts to be considered a campaign
    'focus_on_anomalous': True,       # Only analyze high/critical anomaly posts
    'weight_by_anomaly': True,        # Weight edges by anomaly scores
    'top_k_neighbors': 50,            # Max neighbors per post (for efficiency)
}

print(f"\n📋 Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading prepared data and anomaly results...")

try:
    # Load embeddings and metadata
    prepared_data = torch.load("prepared_clustering_data.pt", 
                              map_location=device, weights_only=False)
    z_out = prepared_data['z_out'].cpu().numpy()
    user_ids = prepared_data['user_ids']
    timestamps = prepared_data['timestamps']
    post_ids = prepared_data['post_ids']
    
    # Load anomaly detection results
    anomaly_results = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")
    
    print(f"✅ Loaded: {len(z_out)} posts")
    print(f"   Embeddings: {z_out.shape}")
    print(f"   Anomaly results: {len(anomaly_results)} rows")
    
except Exception as e:
    print(f"⚠️ Error loading data: {e}")
    print("   Make sure you've run the anomaly detection pipeline first!")
    exit(1)

# ============================================================================
# STEP 2: FILTER TO ANOMALOUS POSTS (OPTIONAL)
# ============================================================================

if CONFIG['focus_on_anomalous']:
    print("\n[STEP 2] Filtering to anomalous posts...")
    
    # Keep only high-risk posts
    anomalous_mask = anomaly_results['anomaly_level'].isin(['high', 'critical'])
    anomalous_indices = anomaly_results[anomalous_mask].index.tolist()
    
    z_out_filtered = z_out[anomalous_indices]
    user_ids_filtered = [user_ids[i] for i in anomalous_indices]
    timestamps_filtered = timestamps[anomalous_indices]
    post_ids_filtered = [post_ids[i] for i in anomalous_indices]
    anomaly_scores_filtered = anomaly_results.loc[anomalous_indices, 'anomaly_score'].values
    
    print(f"✅ Filtered to {len(z_out_filtered)} anomalous posts "
          f"({len(z_out_filtered)/len(z_out)*100:.1f}% of total)")
else:
    print("\n[STEP 2] Using all posts...")
    z_out_filtered = z_out
    user_ids_filtered = user_ids
    timestamps_filtered = timestamps
    post_ids_filtered = post_ids
    anomaly_scores_filtered = anomaly_results['anomaly_score'].values
    print(f"✅ Using all {len(z_out_filtered)} posts")

n_posts = len(z_out_filtered)

# ============================================================================
# STEP 3: COMPUTE CONTENT SIMILARITY MATRIX
# ============================================================================

print("\n[STEP 3] Computing content similarity matrix...")

# Compute cosine similarity between all posts
print(f"   Computing {n_posts}x{n_posts} similarity matrix...")
sim_matrix = cosine_similarity(z_out_filtered)

# Clip negative similarities to 0 (only care about positive similarity)
sim_matrix = np.clip(sim_matrix, 0, 1)

print(f"✅ Similarity matrix computed: {sim_matrix.shape}")
print(f"   Mean similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean():.3f}")
print(f"   Max similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].max():.3f}")

# ============================================================================
# STEP 4: APPLY TIME DECAY
# ============================================================================

print("\n[STEP 4] Applying temporal decay...")

# Convert timestamps to seconds
if isinstance(timestamps_filtered[0], datetime):
    timestamps_sec = np.array([t.timestamp() for t in timestamps_filtered])
elif isinstance(timestamps_filtered[0], (int, float)):
    timestamps_sec = timestamps_filtered.astype(float)
else:
    # Try to parse as datetime
    timestamps_sec = np.array([pd.to_datetime(t).timestamp() for t in timestamps_filtered])

# Compute time difference matrix (in seconds)
print(f"   Computing time differences...")
delta_t = np.abs(timestamps_sec[:, None] - timestamps_sec[None, :])

# Apply exponential time decay
tau = CONFIG['time_decay_tau']
time_weights = np.exp(-delta_t / tau)

# Combine similarity and time decay
W = sim_matrix * time_weights

print(f"✅ Time-decayed weights computed")
print(f"   Time decay factor (tau): {tau}s ({tau/3600:.1f} hours)")
print(f"   Mean weight: {W[np.triu_indices_from(W, k=1)].mean():.3f}")

# ============================================================================
# STEP 5: WEIGHT BY ANOMALY SCORES (OPTIONAL)
# ============================================================================

if CONFIG['weight_by_anomaly']:
    print("\n[STEP 5] Weighting edges by anomaly scores...")
    
    # Normalize anomaly scores to [0.5, 1.5] range
    # (so even low-anomaly posts still have some weight)
    anomaly_normalized = (anomaly_scores_filtered - anomaly_scores_filtered.min()) / \
                        (anomaly_scores_filtered.max() - anomaly_scores_filtered.min() + 1e-10)
    anomaly_weights = 0.5 + anomaly_normalized  # Range: [0.5, 1.5]
    
    # Apply anomaly weights to edges (geometric mean of node weights)
    anomaly_weight_matrix = np.sqrt(anomaly_weights[:, None] * anomaly_weights[None, :])
    W = W * anomaly_weight_matrix
    
    print(f"✅ Applied anomaly weighting")
    print(f"   Anomaly score range: [{anomaly_scores_filtered.min():.3f}, "
          f"{anomaly_scores_filtered.max():.3f}]")
else:
    print("\n[STEP 5] Skipping anomaly weighting...")

# ============================================================================
# STEP 6: CREATE SPARSE GRAPH
# ============================================================================

print("\n[STEP 6] Building post similarity graph...")

# Keep only strong connections (above threshold)
threshold = CONFIG['similarity_threshold']

# Optional: keep only top-k neighbors per post for efficiency
top_k = CONFIG['top_k_neighbors']

print(f"   Applying threshold: {threshold}")
print(f"   Keeping top-{top_k} neighbors per post")

# For each post, keep only top-k strongest edges above threshold
edges = []
weights_list = []

for i in tqdm(range(n_posts), desc="   Building edges"):
    # Get similarities for this post
    row = W[i].copy()
    row[i] = 0  # Remove self-loop
    
    # Keep only above threshold
    valid_indices = np.where(row > threshold)[0]
    
    if len(valid_indices) > 0:
        # Get top-k
        if len(valid_indices) > top_k:
            top_indices = valid_indices[np.argsort(row[valid_indices])[-top_k:]]
        else:
            top_indices = valid_indices
        
        # Add edges
        for j in top_indices:
            if i < j:  # Only add each edge once
                edges.append((i, j))
                weights_list.append(row[j])

print(f"\n✅ Graph constructed:")
print(f"   Nodes: {n_posts}")
print(f"   Edges: {len(edges)}")
print(f"   Density: {len(edges) / (n_posts * (n_posts-1) / 2) * 100:.2f}%")

# Create NetworkX graph
G = nx.Graph()
G.add_nodes_from(range(n_posts))
for (i, j), weight in zip(edges, weights_list):
    G.add_edge(i, j, weight=weight)

# Graph statistics
print(f"\n   Graph Statistics:")
print(f"      Connected components: {nx.number_connected_components(G)}")
print(f"      Average degree: {sum(dict(G.degree()).values()) / n_posts:.2f}")
print(f"      Average clustering coefficient: {nx.average_clustering(G):.3f}")

# ============================================================================
# STEP 7: COMMUNITY DETECTION (CAMPAIGN DETECTION)
# ============================================================================

print("\n[STEP 7] Detecting campaigns via community detection...")

# Apply Louvain algorithm
print("   Running Louvain community detection...")
partition = community_louvain.best_partition(G, weight='weight', random_state=42)

# Get modularity score
modularity = community_louvain.modularity(partition, G, weight='weight')

print(f"✅ Community detection complete")
print(f"   Modularity: {modularity:.3f} (higher = better community structure)")
print(f"   Communities found: {len(set(partition.values()))}")

# ============================================================================
# STEP 8: ANALYZE CAMPAIGNS
# ============================================================================

print("\n[STEP 8] Analyzing detected campaigns...")

# Group posts by campaign
campaigns = defaultdict(list)
for node_id, campaign_id in partition.items():
    campaigns[campaign_id].append(node_id)

# Filter to significant campaigns (>= min_size)
min_size = CONFIG['min_campaign_size']
significant_campaigns = {
    cid: nodes for cid, nodes in campaigns.items() 
    if len(nodes) >= min_size
}

print(f"\n✅ Campaign Analysis:")
print(f"   Total communities: {len(campaigns)}")
print(f"   Significant campaigns (≥{min_size} posts): {len(significant_campaigns)}")

# Compute campaign statistics
campaign_stats = []

for campaign_id, node_indices in significant_campaigns.items():
    # Get posts in this campaign
    campaign_posts = [post_ids_filtered[i] for i in node_indices]
    campaign_users = [user_ids_filtered[i] for i in node_indices]
    campaign_timestamps = [timestamps_filtered[i] for i in node_indices]
    campaign_anomaly_scores = [anomaly_scores_filtered[i] for i in node_indices]
    
    # Time statistics
    if isinstance(campaign_timestamps[0], datetime):
        time_span = max(campaign_timestamps) - min(campaign_timestamps)
        time_span_hours = time_span.total_seconds() / 3600
    else:
        time_span_sec = max(campaign_timestamps) - min(campaign_timestamps)
        time_span_hours = time_span_sec / 3600
    
    # User statistics
    unique_users = len(set(campaign_users))
    posts_per_user = len(campaign_posts) / unique_users
    
    # Anomaly statistics
    mean_anomaly = np.mean(campaign_anomaly_scores)
    max_anomaly = np.max(campaign_anomaly_scores)
    
    # Coordination score (higher = more coordinated)
    # Based on: many posts, few users, short time span, high anomaly
    coordination_score = (
        len(campaign_posts) *          # Volume
        (1 / unique_users) *            # Concentration (fewer users = higher)
        (1 / (time_span_hours + 1)) *  # Speed (faster = higher)
        mean_anomaly * 10               # Suspicion
    )
    
    campaign_stats.append({
        'campaign_id': campaign_id,
        'n_posts': len(campaign_posts),
        'n_users': unique_users,
        'posts_per_user': posts_per_user,
        'time_span_hours': time_span_hours,
        'mean_anomaly_score': mean_anomaly,
        'max_anomaly_score': max_anomaly,
        'coordination_score': coordination_score,
        'post_ids': campaign_posts,
        'user_ids': list(set(campaign_users)),
        'node_indices': node_indices
    })

# Sort by coordination score
campaign_stats_sorted = sorted(campaign_stats, 
                               key=lambda x: x['coordination_score'], 
                               reverse=True)

# Display top campaigns
print(f"\n🚨 Top 10 Most Coordinated Campaigns:")
print(f"{'ID':<6} {'Posts':<8} {'Users':<8} {'P/U':<6} {'Hours':<8} {'Anomaly':<10} {'Coord Score':<12}")
print("-" * 80)

for camp in campaign_stats_sorted[:10]:
    print(f"{camp['campaign_id']:<6} "
          f"{camp['n_posts']:<8} "
          f"{camp['n_users']:<8} "
          f"{camp['posts_per_user']:<6.1f} "
          f"{camp['time_span_hours']:<8.1f} "
          f"{camp['mean_anomaly_score']:<10.3f} "
          f"{camp['coordination_score']:<12.2f}")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n[STEP 9] Saving campaign detection results...")

output_dir = Path("campaign_detection_results")
output_dir.mkdir(exist_ok=True)

# Save campaign assignments
campaign_assignments = []
for node_id, campaign_id in partition.items():
    campaign_assignments.append({
        'post_id': post_ids_filtered[node_id],
        'user_id': user_ids_filtered[node_id],
        'timestamp': timestamps_filtered[node_id],
        'campaign_id': campaign_id,
        'anomaly_score': anomaly_scores_filtered[node_id]
    })

campaign_df = pd.DataFrame(campaign_assignments)
campaign_df.to_csv(output_dir / "campaign_assignments.csv", index=False)

# Save campaign statistics
campaign_stats_df = pd.DataFrame([
    {k: v for k, v in camp.items() if k not in ['post_ids', 'user_ids', 'node_indices']}
    for camp in campaign_stats_sorted
])
campaign_stats_df.to_csv(output_dir / "campaign_statistics.csv", index=False)

# Save detailed campaign reports
for camp in campaign_stats_sorted[:20]:  # Top 20 campaigns
    campaign_id = camp['campaign_id']
    campaign_report_dir = output_dir / f"campaign_{campaign_id}"
    campaign_report_dir.mkdir(exist_ok=True)
    
    # Save post list
    pd.DataFrame({
        'post_id': camp['post_ids'],
    }).to_csv(campaign_report_dir / "posts.csv", index=False)
    
    # Save user list
    pd.DataFrame({
        'user_id': camp['user_ids'],
    }).to_csv(campaign_report_dir / "users.csv", index=False)

# Save graph
nx.write_gpickle(G, output_dir / "post_similarity_graph.gpickle")

print(f"✅ Results saved to {output_dir}/")

# ============================================================================
# STEP 10: VISUALIZATION
# ============================================================================

print("\n[STEP 10] Creating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Campaign size distribution
ax1 = fig.add_subplot(gs[0, 0])
campaign_sizes = [len(nodes) for nodes in campaigns.values()]
ax1.hist(campaign_sizes, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(min_size, color='red', linestyle='--', label=f'Min size threshold')
ax1.set_title('Campaign Size Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Posts per Campaign')
ax1.set_ylabel('Frequency')
ax1.set_yscale('log')
ax1.legend()

# 2. Top campaigns by coordination score
ax2 = fig.add_subplot(gs[0, 1])
top_10 = campaign_stats_sorted[:10]
ax2.barh([str(c['campaign_id']) for c in top_10], 
         [c['coordination_score'] for c in top_10],
         color='coral', edgecolor='black')
ax2.set_title('Top 10 Campaigns by Coordination Score', fontsize=12, fontweight='bold')
ax2.set_xlabel('Coordination Score')
ax2.set_ylabel('Campaign ID')
ax2.invert_yaxis()

# 3. Posts vs Users scatter
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter([c['n_users'] for c in campaign_stats],
           [c['n_posts'] for c in campaign_stats],
           c=[c['mean_anomaly_score'] for c in campaign_stats],
           cmap='YlOrRd', s=50, alpha=0.6, edgecolors='black')
ax3.set_title('Campaign Profile: Posts vs Users', fontsize=12, fontweight='bold')
ax3.set_xlabel('Unique Users')
ax3.set_ylabel('Total Posts')
ax3.set_xscale('log')
ax3.set_yscale('log')
cbar = plt.colorbar(ax3.collections[0], ax=ax3)
cbar.set_label('Mean Anomaly Score')

# 4. Time span distribution
ax4 = fig.add_subplot(gs[1, 0])
time_spans = [c['time_span_hours'] for c in campaign_stats if c['time_span_hours'] < 168]  # <1 week
ax4.hist(time_spans, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
ax4.set_title('Campaign Duration (<1 week)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Time Span (hours)')
ax4.set_ylabel('Frequency')

# 5. Posts per user distribution
ax5 = fig.add_subplot(gs[1, 1])
posts_per_user = [c['posts_per_user'] for c in campaign_stats]
ax5.hist(posts_per_user, bins=50, color='plum', edgecolor='black', alpha=0.7)
ax5.set_title('Posts per User in Campaigns', fontsize=12, fontweight='bold')
ax5.set_xlabel('Posts per User')
ax5.set_ylabel('Frequency')

# 6. Anomaly score vs campaign size
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter([c['n_posts'] for c in campaign_stats],
           [c['mean_anomaly_score'] for c in campaign_stats],
           s=50, alpha=0.6, color='orange', edgecolors='black')
ax6.set_title('Campaign Size vs Anomaly Score', fontsize=12, fontweight='bold')
ax6.set_xlabel('Posts in Campaign')
ax6.set_ylabel('Mean Anomaly Score')
ax6.set_xscale('log')

# 7. Load UMAP if available
try:
    anomaly_models = torch.load("anomaly_detection_results/anomaly_models.pt", 
                               map_location='cpu', weights_only=False)
    X_2d_full = anomaly_models['X_2d']
    
    # Get UMAP coords for filtered posts
    if CONFIG['focus_on_anomalous']:
        X_2d = X_2d_full[anomalous_indices]
    else:
        X_2d = X_2d_full
    
    # Color by campaign
    ax7 = fig.add_subplot(gs[2, :2])
    campaign_colors = [partition.get(i, -1) for i in range(len(X_2d))]
    scatter = ax7.scatter(X_2d[:, 0], X_2d[:, 1], 
                         c=campaign_colors, cmap='tab20', 
                         s=15, alpha=0.6, edgecolors='none')
    ax7.set_title('UMAP: Posts Colored by Campaign', fontsize=14, fontweight='bold')
    ax7.set_xlabel('UMAP 1')
    ax7.set_ylabel('UMAP 2')
    
    # Highlight top 3 campaigns
    for i, camp in enumerate(campaign_stats_sorted[:3]):
        nodes = camp['node_indices']
        ax7.scatter(X_2d[nodes, 0], X_2d[nodes, 1], 
                   s=100, marker='*', linewidths=2,
                   edgecolors='red', facecolors='none',
                   label=f"Campaign {camp['campaign_id']} (n={camp['n_posts']})")
    ax7.legend(loc='best')
    
except:
    print("   ⚠️  Could not load UMAP visualization")

# 8. Network degree distribution
ax8 = fig.add_subplot(gs[2, 2])
degrees = [G.degree(n) for n in G.nodes()]
ax8.hist(degrees, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax8.set_title('Network Degree Distribution', fontsize=12, fontweight='bold')
ax8.set_xlabel('Node Degree')
ax8.set_ylabel('Frequency')
ax8.set_yscale('log')

plt.savefig(output_dir / "campaign_analysis.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization")

print("\n" + "="*80)
print("✅ CAMPAIGN DETECTION COMPLETE!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Total posts analyzed: {n_posts}")
print(f"   Communities detected: {len(campaigns)}")
print(f"   Significant campaigns: {len(significant_campaigns)}")
print(f"   Graph modularity: {modularity:.3f}")
print(f"\n   Top Campaign:")
if campaign_stats_sorted:
    top = campaign_stats_sorted[0]
    print(f"      ID: {top['campaign_id']}")
    print(f"      Posts: {top['n_posts']}")
    print(f"      Users: {top['n_users']}")
    print(f"      Duration: {top['time_span_hours']:.1f} hours")
    print(f"      Coordination score: {top['coordination_score']:.2f}")

print(f"\n   Output files:")
print(f"      📄 campaign_assignments.csv - Post-to-campaign mapping")
print(f"      📊 campaign_statistics.csv - Campaign-level metrics")
print(f"      📁 campaign_X/ - Detailed reports for top campaigns")
print(f"      🔗 post_similarity_graph.gpickle - NetworkX graph object")
print(f"      📊 campaign_analysis.png - Comprehensive visualization")
print(f"\n   Results saved to: {output_dir}/")