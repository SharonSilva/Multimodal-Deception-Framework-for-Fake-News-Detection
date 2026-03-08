"""
BUILD HETEROGENEOUS GRAPH FOR TEMPORAL GNN
==========================================
Constructs a heterogeneous graph from existing pipeline outputs.

Node types:
  - post            : each post (features = z_out + v_mismatch from emotion model)
  - user            : each unique user (features = aggregated post embeddings + metadata)
  - hashtag         : each unique hashtag (features = averaged post embeddings)
  - community       : cluster groups from cluster_label column
  - deception_cluster: anomaly severity groups (critical/high)

Edge types:
  - (user, creates, post)
  - (post, contains, hashtag)
  - (hashtag, cooccurs_with, hashtag)
  - (user, interacts_with, user)       via shared hashtags/mentions
  - (post, belongs_to, community)
  - (post, flagged_in, deception_cluster)
  - (deception_cluster, colludes_with, deception_cluster)

Outputs saved to heterogeneous_graph/:
  - node_features.pt      {node_type: tensor}
  - edge_dict.pt          {(src, rel, dst): (edge_index, edge_weight)}
  - node_mappings.pkl     {node_type: {original_id: local_idx}}
  - graph_stats.txt       summary of graph structure
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import ast
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUILDING HETEROGENEOUS GRAPH FOR TEMPORAL GNN")
print("=" * 80)

output_dir = Path("heterogeneous_graph")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD ALL DATA SOURCES
# ============================================================================
print("\n[STEP 1] Loading data sources...")

# Raw dataset
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)
print(f"   ✅ Dataset: {len(df)} posts, {df['user_id'].nunique()} users")

# Emotion-aware embeddings
cluster_data = torch.load("prepared_clustering_data.pt", map_location='cpu', weights_only=False)
z_out      = cluster_data['z_out']        # [N, 128]
v_mismatch = cluster_data['v_mismatch']   # [N, 128]
post_ids_emb  = [str(p) for p in cluster_data['post_ids']]
user_ids_emb  = cluster_data['user_ids']
timestamps_emb = cluster_data['timestamps']
print(f"   ✅ Embeddings: z_out={z_out.shape}, v_mismatch={v_mismatch.shape}")

# Anomaly assignments
anomaly_df = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")
anomaly_df['post_id'] = anomaly_df['post_id'].astype(str)
anomaly_df['user_id'] = anomaly_df['user_id'].astype(str)
print(f"   ✅ Anomaly assignments: {len(anomaly_df)} posts")

# Build post_id → embedding index lookup
post_id_to_emb_idx = {pid: i for i, pid in enumerate(post_ids_emb)}

# ============================================================================
# STEP 2: PARSE HASHTAGS AND MENTIONS
# ============================================================================
print("\n[STEP 2] Parsing hashtags and mentions...")

def safe_parse_list(val):
    """Safely parse a list-like column that may be stored as string."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == '' or val == 'nan':
        return []
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else []
    except:
        return []

df['hashtags_parsed'] = df['hashtags'].apply(safe_parse_list)
df['mentions_parsed'] = df['mentions'].apply(safe_parse_list)

# Normalise hashtags to lowercase without #
def normalise_hashtag(h):
    return str(h).lower().lstrip('#').strip()

df['hashtags_parsed'] = df['hashtags_parsed'].apply(
    lambda lst: [normalise_hashtag(h) for h in lst if str(h).strip()]
)
df['mentions_parsed'] = df['mentions_parsed'].apply(
    lambda lst: [str(m).lstrip('@').strip().lower() for m in lst if str(m).strip()]
)

all_hashtags = set()
for lst in df['hashtags_parsed']:
    all_hashtags.update(lst)
all_hashtags = sorted(all_hashtags)
print(f"   ✅ Unique hashtags: {len(all_hashtags)}")

# ============================================================================
# STEP 3: BUILD NODE MAPPINGS
# ============================================================================
print("\n[STEP 3] Building node mappings...")

# --- POST nodes (only posts that have embeddings) ---
post_nodes = [pid for pid in post_ids_emb if pid in df['post_id'].values]
post_to_idx = {pid: i for i, pid in enumerate(post_nodes)}
print(f"   Posts:              {len(post_to_idx)}")

# --- USER nodes ---
all_users = sorted(df['user_id'].unique())
user_to_idx = {uid: i for i, uid in enumerate(all_users)}
print(f"   Users:              {len(user_to_idx)}")

# --- HASHTAG nodes ---
hashtag_to_idx = {h: i for i, h in enumerate(all_hashtags)}
print(f"   Hashtags:           {len(hashtag_to_idx)}")

# --- COMMUNITY nodes (from cluster_label) ---
community_ids = sorted(df['cluster_label'].dropna().unique().astype(int))
community_to_idx = {cid: i for i, cid in enumerate(community_ids)}
print(f"   Communities:        {len(community_to_idx)}")

# --- DECEPTION CLUSTER nodes (anomaly severity groups) ---
# Group: 0=high_anomaly (critical+high), 1=medium, 2=low_anomaly
# We create one deception cluster node per unique (user risk group) combination
# to represent coordinated behaviour groups
deception_levels = ['critical', 'high']
deception_posts = anomaly_df[anomaly_df['anomaly_level'].isin(deception_levels)].copy()

# Group deception clusters by user — each user gets assigned to a cluster
# based on their dominant anomaly pattern
user_deception_map = {}
for user_id, group in deception_posts.groupby('user_id'):
    dominant = group['anomaly_level'].mode()[0]
    user_deception_map[user_id] = dominant

# Create deception cluster nodes: one per unique (anomaly_level) × (community)
deception_cluster_ids = []
post_to_deception = {}  # post_id → deception_cluster_local_idx

deception_posts_with_community = deception_posts.merge(
    df[['post_id', 'cluster_label']], on='post_id', how='left'
)

dc_key_to_idx = {}
for _, row in deception_posts_with_community.iterrows():
    key = (str(row['anomaly_level']), int(row['cluster_label']) if not pd.isna(row['cluster_label']) else -1)
    if key not in dc_key_to_idx:
        dc_key_to_idx[key] = len(dc_key_to_idx)
    post_to_deception[str(row['post_id'])] = dc_key_to_idx[key]

deception_cluster_to_idx = dc_key_to_idx
n_deception_clusters = len(deception_cluster_to_idx)
print(f"   Deception clusters: {n_deception_clusters}")

node_mappings = {
    'post':               post_to_idx,
    'user':               user_to_idx,
    'hashtag':            hashtag_to_idx,
    'community':          community_to_idx,
    'deception_cluster':  deception_cluster_to_idx
}

# ============================================================================
# STEP 4: BUILD NODE FEATURES
# ============================================================================
print("\n[STEP 4] Building node features...")

# --- POST features: z_out concatenated with v_mismatch → [N_posts, 256] ---
post_feat_list = []
for pid in post_nodes:
    if pid in post_id_to_emb_idx:
        idx = post_id_to_emb_idx[pid]
        feat = torch.cat([z_out[idx], v_mismatch[idx]], dim=0)  # [256]
    else:
        feat = torch.zeros(256)
    post_feat_list.append(feat)
post_features = torch.stack(post_feat_list)  # [N_posts, 256]
print(f"   Post features:     {post_features.shape}")

# --- USER features: mean of post embeddings + metadata features ---
user_feat_dim = 256 + 6  # mean post embedding + 6 metadata features
user_feat_list = []

# Build user → post indices lookup
user_post_lists = defaultdict(list)
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) == 0:
        continue
    uid = str(row.iloc[0]['user_id'])
    user_post_lists[uid].append(i)

for uid in all_users:
    # Mean embedding of user's posts
    if uid in user_post_lists and len(user_post_lists[uid]) > 0:
        idxs = user_post_lists[uid]
        mean_emb = post_features[idxs].mean(dim=0)  # [256]
    else:
        mean_emb = torch.zeros(256)

    # Metadata features
    user_posts = df[df['user_id'] == uid]
    if len(user_posts) > 0:
        n_posts       = float(len(user_posts))
        n_hashtags    = float(user_posts['hashtags_count'].mean())
        n_mentions    = float(user_posts['user_mentions_count'].mean())
        n_urls        = float(user_posts['urls_count'].mean())
        n_emojis      = float(user_posts['emojis_count'].mean())
        contradiction = float(user_posts['contradiction_score'].mean())
        meta_feat = torch.tensor([n_posts, n_hashtags, n_mentions,
                                   n_urls, n_emojis, contradiction], dtype=torch.float32)
    else:
        meta_feat = torch.zeros(6)

    user_feat_list.append(torch.cat([mean_emb, meta_feat]))

user_features = torch.stack(user_feat_list)  # [N_users, 262]
print(f"   User features:     {user_features.shape}")

# --- HASHTAG features: mean embedding of posts containing each hashtag ---
hashtag_post_map = defaultdict(list)
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) == 0:
        continue
    for h in row.iloc[0]['hashtags_parsed']:
        if h in hashtag_to_idx:
            hashtag_post_map[h].append(i)

hashtag_feat_list = []
for h in all_hashtags:
    if h in hashtag_post_map and len(hashtag_post_map[h]) > 0:
        idxs = hashtag_post_map[h]
        feat = post_features[idxs].mean(dim=0)  # [256]
    else:
        feat = torch.zeros(256)
    hashtag_feat_list.append(feat)
hashtag_features = torch.stack(hashtag_feat_list)  # [N_hashtags, 256]
print(f"   Hashtag features:  {hashtag_features.shape}")

# --- COMMUNITY features: mean embedding of posts in each community ---
community_post_map = defaultdict(list)
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) == 0:
        continue
    cl = row.iloc[0]['cluster_label']
    if not pd.isna(cl):
        community_post_map[int(cl)].append(i)

community_feat_list = []
for cid in community_ids:
    if cid in community_post_map and len(community_post_map[cid]) > 0:
        idxs = community_post_map[cid]
        feat = post_features[idxs].mean(dim=0)
    else:
        feat = torch.zeros(256)
    community_feat_list.append(feat)
community_features = torch.stack(community_feat_list)  # [N_communities, 256]
print(f"   Community features:{community_features.shape}")

# --- DECEPTION CLUSTER features: mean embedding of posts in each cluster ---
dc_post_map = defaultdict(list)
for i, pid in enumerate(post_nodes):
    if pid in post_to_deception:
        dc_idx = post_to_deception[pid]
        dc_post_map[dc_idx].append(i)

dc_feat_list = []
for dc_idx in range(n_deception_clusters):
    if dc_idx in dc_post_map and len(dc_post_map[dc_idx]) > 0:
        idxs = dc_post_map[dc_idx]
        # Also include anomaly score as extra feature
        anomaly_scores_dc = []
        for pidx in idxs:
            pid = post_nodes[pidx]
            row = anomaly_df[anomaly_df['post_id'] == pid]
            if len(row) > 0:
                anomaly_scores_dc.append(row.iloc[0]['anomaly_score'])
        mean_anomaly = float(np.mean(anomaly_scores_dc)) if anomaly_scores_dc else 0.0
        feat = post_features[idxs].mean(dim=0)
    else:
        feat = torch.zeros(256)
    dc_feat_list.append(feat)
dc_features = torch.stack(dc_feat_list)  # [N_dc, 256]
print(f"   Deception cluster: {dc_features.shape}")

node_features = {
    'post':              post_features,
    'user':              user_features,
    'hashtag':           hashtag_features,
    'community':         community_features,
    'deception_cluster': dc_features
}

# ============================================================================
# STEP 5: BUILD EDGES
# ============================================================================
print("\n[STEP 5] Building edges...")

edge_dict = {}

# Helper to create edge tensor
def make_edge(src_list, dst_list, weight_list=None):
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    if weight_list is None:
        weights = torch.ones(len(src_list), dtype=torch.float32)
    else:
        weights = torch.tensor(weight_list, dtype=torch.float32)
    return edge_index, weights

# ── Edge 1: user CREATES post ─────────────────────────────────────────────
print("   Building user → creates → post edges...")
uc_src, uc_dst = [], []
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) == 0:
        continue
    uid = str(row.iloc[0]['user_id'])
    if uid in user_to_idx:
        uc_src.append(user_to_idx[uid])
        uc_dst.append(i)

edge_dict[('user', 'creates', 'post')] = make_edge(uc_src, uc_dst)
print(f"   ✅ user→creates→post: {len(uc_src)} edges")

# ── Edge 2: post CONTAINS hashtag ─────────────────────────────────────────
print("   Building post → contains → hashtag edges...")
ph_src, ph_dst = [], []
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) == 0:
        continue
    for h in row.iloc[0]['hashtags_parsed']:
        if h in hashtag_to_idx:
            ph_src.append(i)
            ph_dst.append(hashtag_to_idx[h])

edge_dict[('post', 'contains', 'hashtag')] = make_edge(ph_src, ph_dst)
print(f"   ✅ post→contains→hashtag: {len(ph_src)} edges")

# ── Edge 3: hashtag CO-OCCURS WITH hashtag ────────────────────────────────
print("   Building hashtag → cooccurs_with → hashtag edges...")
hh_src, hh_dst, hh_weights = [], [], []
cooccur_counts = defaultdict(int)

for _, row in df.iterrows():
    hashtags = row['hashtags_parsed']
    for i_h in range(len(hashtags)):
        for j_h in range(i_h + 1, len(hashtags)):
            h1, h2 = hashtags[i_h], hashtags[j_h]
            if h1 in hashtag_to_idx and h2 in hashtag_to_idx:
                key = (min(h1, h2), max(h1, h2))
                cooccur_counts[key] += 1

# Only keep co-occurrences that happen more than once (reduce noise)
for (h1, h2), count in cooccur_counts.items():
    if count >= 2:
        idx1, idx2 = hashtag_to_idx[h1], hashtag_to_idx[h2]
        hh_src.extend([idx1, idx2])
        hh_dst.extend([idx2, idx1])
        hh_weights.extend([float(count), float(count)])

if len(hh_src) > 0:
    edge_dict[('hashtag', 'cooccurs_with', 'hashtag')] = make_edge(hh_src, hh_dst, hh_weights)
    print(f"   ✅ hashtag→cooccurs_with→hashtag: {len(hh_src)} edges")
else:
    print(f"   ⚠️  No hashtag co-occurrences found (all unique)")

# ── Edge 4: user INTERACTS WITH user (via shared hashtags/mentions) ────────
print("   Building user → interacts_with → user edges...")
uu_src, uu_dst, uu_weights = [], [], []
interaction_counts = defaultdict(int)

# Via shared hashtags
hashtag_users = defaultdict(set)
for _, row in df.iterrows():
    uid = str(row['user_id'])
    for h in row['hashtags_parsed']:
        hashtag_users[h].add(uid)

for h, users in hashtag_users.items():
    users = list(users)
    for i_u in range(len(users)):
        for j_u in range(i_u + 1, len(users)):
            u1, u2 = users[i_u], users[j_u]
            if u1 in user_to_idx and u2 in user_to_idx:
                key = (min(u1, u2), max(u1, u2))
                interaction_counts[key] += 1

# Via mentions
for _, row in df.iterrows():
    uid = str(row['user_id'])
    for mention in row['mentions_parsed']:
        # Find if mentioned user exists in our dataset
        mentioned_rows = df[df['username'].str.lower() == mention.lower()]
        if len(mentioned_rows) > 0:
            mentioned_uid = str(mentioned_rows.iloc[0]['user_id'])
            if uid in user_to_idx and mentioned_uid in user_to_idx and uid != mentioned_uid:
                key = (min(uid, mentioned_uid), max(uid, mentioned_uid))
                interaction_counts[key] += 1

# Only keep interactions that happen at least twice
for (u1, u2), count in interaction_counts.items():
    if count >= 2:
        idx1, idx2 = user_to_idx[u1], user_to_idx[u2]
        uu_src.extend([idx1, idx2])
        uu_dst.extend([idx2, idx1])
        uu_weights.extend([float(count), float(count)])

if len(uu_src) > 0:
    edge_dict[('user', 'interacts_with', 'user')] = make_edge(uu_src, uu_dst, uu_weights)
    print(f"   ✅ user→interacts_with→user: {len(uu_src)} edges")
else:
    # Create sparse fallback — users who posted same hashtag at least once
    print("   ⚠️  Few interactions found, using single-occurrence threshold...")
    for (u1, u2), count in interaction_counts.items():
        idx1, idx2 = user_to_idx[u1], user_to_idx[u2]
        uu_src.extend([idx1, idx2])
        uu_dst.extend([idx2, idx1])
        uu_weights.extend([1.0, 1.0])
    if len(uu_src) > 0:
        edge_dict[('user', 'interacts_with', 'user')] = make_edge(uu_src, uu_dst, uu_weights)
        print(f"   ✅ user→interacts_with→user: {len(uu_src)} edges (single-occurrence)")
    else:
        print(f"   ⚠️  No user interactions found — skipping this edge type")

# ── Edge 5: post BELONGS TO community ─────────────────────────────────────
print("   Building post → belongs_to → community edges...")
pc_src, pc_dst = [], []
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) == 0:
        continue
    cl = row.iloc[0]['cluster_label']
    if not pd.isna(cl) and int(cl) in community_to_idx:
        pc_src.append(i)
        pc_dst.append(community_to_idx[int(cl)])

edge_dict[('post', 'belongs_to', 'community')] = make_edge(pc_src, pc_dst)
print(f"   ✅ post→belongs_to→community: {len(pc_src)} edges")

# ── Edge 6: post FLAGGED IN deception_cluster ──────────────────────────────
print("   Building post → flagged_in → deception_cluster edges...")
pd_src, pd_dst = [], []
for i, pid in enumerate(post_nodes):
    if pid in post_to_deception:
        pd_src.append(i)
        pd_dst.append(post_to_deception[pid])

if len(pd_src) > 0:
    edge_dict[('post', 'flagged_in', 'deception_cluster')] = make_edge(pd_src, pd_dst)
    print(f"   ✅ post→flagged_in→deception_cluster: {len(pd_src)} edges")
else:
    print(f"   ⚠️  No deception cluster assignments found")

# ── Edge 7: deception_cluster COLLUDES WITH deception_cluster ──────────────
print("   Building deception_cluster → colludes_with → deception_cluster edges...")
dcd_src, dcd_dst = [], []

# Two deception clusters collude if they share users
dc_to_users = defaultdict(set)
for pid, dc_idx in post_to_deception.items():
    row = df[df['post_id'] == pid]
    if len(row) > 0:
        uid = str(row.iloc[0]['user_id'])
        dc_to_users[dc_idx].add(uid)

dc_indices = list(range(n_deception_clusters))
for i in range(len(dc_indices)):
    for j in range(i + 1, len(dc_indices)):
        shared = dc_to_users[i] & dc_to_users[j]
        if len(shared) > 0:
            dcd_src.extend([i, j])
            dcd_dst.extend([j, i])

if len(dcd_src) > 0:
    edge_dict[('deception_cluster', 'colludes_with', 'deception_cluster')] = \
        make_edge(dcd_src, dcd_dst)
    print(f"   ✅ deception_cluster→colludes_with→deception_cluster: {len(dcd_src)} edges")
else:
    print(f"   ⚠️  No colluding deception clusters found")

# ============================================================================
# STEP 6: VALIDATE GRAPH
# ============================================================================
print("\n[STEP 6] Validating graph...")

all_valid = True
for etype, (edge_index, edge_weight) in edge_dict.items():
    src_type, rel, dst_type = etype
    n_src = node_features[src_type].shape[0]
    n_dst = node_features[dst_type].shape[0]

    max_src = edge_index[0].max().item() if edge_index.shape[1] > 0 else -1
    max_dst = edge_index[1].max().item() if edge_index.shape[1] > 0 else -1

    valid = (max_src < n_src) and (max_dst < n_dst)
    status = "✅" if valid else "❌"
    if not valid:
        all_valid = False
    print(f"   {status} {src_type}→{rel}→{dst_type}: "
          f"{edge_index.shape[1]} edges | "
          f"max_src={max_src}/{n_src-1} | max_dst={max_dst}/{n_dst-1}")

if all_valid:
    print("\n   ✅ All edge indices are valid")
else:
    print("\n   ❌ Some edges have out-of-bounds indices — fixing...")
    for etype in list(edge_dict.keys()):
        src_type, rel, dst_type = etype
        edge_index, edge_weight = edge_dict[etype]
        n_src = node_features[src_type].shape[0]
        n_dst = node_features[dst_type].shape[0]
        valid_mask = (
            (edge_index[0] >= 0) & (edge_index[0] < n_src) &
            (edge_index[1] >= 0) & (edge_index[1] < n_dst)
        )
        edge_dict[etype] = (edge_index[:, valid_mask], edge_weight[valid_mask])
    print("   ✅ Fixed")

# ============================================================================
# STEP 7: SAVE GRAPH
# ============================================================================
print("\n[STEP 7] Saving graph...")

torch.save(node_features, output_dir / "node_features.pt")
print(f"   ✅ Saved node_features.pt")

torch.save(edge_dict, output_dir / "edge_dict.pt")
print(f"   ✅ Saved edge_dict.pt")

with open(output_dir / "node_mappings.pkl", "wb") as f:
    pickle.dump(node_mappings, f)
print(f"   ✅ Saved node_mappings.pkl")

# Also save timestamps per node type for temporal GNN
post_timestamps = torch.zeros(len(post_nodes), dtype=torch.float32)
for i, pid in enumerate(post_nodes):
    row = anomaly_df[anomaly_df['post_id'] == pid]
    if len(row) > 0:
        post_timestamps[i] = float(row.iloc[0]['timestamp'])

user_timestamps = torch.zeros(len(all_users), dtype=torch.float32)
for uid, idx in user_to_idx.items():
    user_posts = anomaly_df[anomaly_df['user_id'] == uid]
    if len(user_posts) > 0:
        user_timestamps[idx] = float(user_posts['timestamp'].max())

timestamps_dict = {
    'post': post_timestamps,
    'user': user_timestamps,
    'hashtag': torch.zeros(len(all_hashtags), dtype=torch.float32),
    'community': torch.zeros(len(community_ids), dtype=torch.float32),
    'deception_cluster': torch.zeros(n_deception_clusters, dtype=torch.float32)
}
torch.save(timestamps_dict, output_dir / "node_timestamps.pt")
print(f"   ✅ Saved node_timestamps.pt")

# Save post labels for training
post_labels = torch.zeros(len(post_nodes), dtype=torch.long)
label_map = {'fake': 1, 'real': 0}
for i, pid in enumerate(post_nodes):
    row = df[df['post_id'] == pid]
    if len(row) > 0:
        lbl = row.iloc[0]['label']
        post_labels[i] = label_map.get(str(lbl).lower(), 0)

torch.save(post_labels, output_dir / "post_labels.pt")
print(f"   ✅ Saved post_labels.pt")
print(f"      Fake: {post_labels.sum().item()}, Real: {(post_labels == 0).sum().item()}")

# ============================================================================
# STEP 8: PRINT SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ HETEROGENEOUS GRAPH BUILT SUCCESSFULLY")
print("=" * 80)

total_nodes = sum(v.shape[0] for v in node_features.values())
total_edges = sum(ei.shape[1] for ei, _ in edge_dict.values())

summary = f"""
Graph Statistics:
{'='*50}
Node Types:
"""
for ntype, feats in node_features.items():
    summary += f"  {ntype:20s}: {feats.shape[0]:6d} nodes, {feats.shape[1]}D features\n"

summary += f"\nEdge Types:\n"
for (src, rel, dst), (ei, ew) in edge_dict.items():
    summary += f"  {src}→{rel}→{dst}: {ei.shape[1]} edges\n"

summary += f"""
Totals:
  Total nodes: {total_nodes:,}
  Total edges: {total_edges:,}
  Node types:  {len(node_features)}
  Edge types:  {len(edge_dict)}

Output files:
  heterogeneous_graph/node_features.pt
  heterogeneous_graph/edge_dict.pt
  heterogeneous_graph/node_mappings.pkl
  heterogeneous_graph/node_timestamps.pt
  heterogeneous_graph/post_labels.pt
"""

print(summary)

# Save stats to text file
with open(output_dir / "graph_stats.txt", "w") as f:
    f.write(summary)
print("   ✅ Saved graph_stats.txt")
print(f"\n   Next step: python3 train_temporal_hetero_gnn.py")