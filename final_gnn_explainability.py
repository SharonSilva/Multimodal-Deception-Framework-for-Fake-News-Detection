"""
EXPLAINABILITY LAYER — FIXED VERSION
=====================================
Fixes three issues from previous run:

  Fix 1 — VAD alignment: use cluster post_ids as alignment key, not df row index
  Fix 2 — Method gradient: recompute flags from anomaly_assignments.csv using
           75th percentile thresholds on real posts only (same as bridge script)
  Fix 3 — Fusion weights: use correct checkpoint key prefix 'fusion_layer.'
           and correct classifier indices [0,3,6,9]

Outputs saved to explainability_results/:
  fusion_weights_all_posts.csv
  mismatch_analysis.png
  fusion_weight_analysis.png
  dominant_modality.png
  text_attribution.png
  gnn_embedding_umap.png
  method_agreement_gradient.png
  vad_emotion_analysis.png
  case_studies.csv
  case_studies_report.txt
  explainability_summary.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXPLAINABILITY LAYER — FIXED VERSION")
print("=" * 80)

output_dir = Path("explainability_results")
output_dir.mkdir(exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================================
# STEP 1: LOAD ALL DATA — aligned on cluster post_ids
# ============================================================================
print("\n[STEP 1] Loading data (aligned on cluster post_ids)...")

# Cluster data is the alignment anchor — 10,826 posts in consistent order
cluster_data = torch.load("prepared_clustering_data.pt", map_location='cpu', weights_only=False)
z_out        = cluster_data['z_out']       # [10826, 128]
v_mismatch   = cluster_data['v_mismatch']  # [10826, 128]
post_ids_ordered = [str(p) for p in cluster_data['post_ids']]
N = len(post_ids_ordered)
post_id_to_cluster_idx = {pid: i for i, pid in enumerate(post_ids_ordered)}
print(f"   ✅ Cluster embeddings: {N} posts")

# Raw dataset — merge by post_id, not by row index
df_raw = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df_raw['post_id'] = df_raw['post_id'].astype(str)
df_raw = df_raw.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)

# Build aligned dataframe in cluster order
df = pd.DataFrame({'post_id': post_ids_ordered})
df = df.merge(df_raw, on='post_id', how='left')
df['label_binary'] = (df['label'].str.lower() == 'fake').astype(int)
print(f"   ✅ Aligned dataset: {len(df)} posts ({df['label_binary'].sum()} fake)")

# VAD data — also aligned to cluster order via prepared_clustering_data
# The cluster data was built from the same pipeline as VAD data
# but VAD has 13,072 rows (pre-dedup). We must align by post_id.
vad_raw = torch.load("Dataset/twitter/prepared_vad_data.pt")

# Find which VAD rows correspond to our 10,826 cluster posts
# VAD was built from raw df (11,844 rows before any dedup)
# We align using df_raw row order before our dedup
df_for_vad = df_raw.copy()  # 10,826 deduped posts

# VAD is aligned to the RAW df (13,072) — find positions of our post_ids
df_all_raw = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df_all_raw['post_id'] = df_all_raw['post_id'].astype(str)
df_all_raw = df_all_raw.reset_index(drop=True)  # keep duplicates, preserve VAD alignment

# For each cluster post_id, find its first occurrence in df_all_raw
vad_indices = []
pid_to_raw_idx = {}
for idx, pid in enumerate(df_all_raw['post_id']):
    if pid not in pid_to_raw_idx:
        pid_to_raw_idx[pid] = idx

for pid in post_ids_ordered:
    if pid in pid_to_raw_idx:
        vad_indices.append(pid_to_raw_idx[pid])
    else:
        vad_indices.append(None)

# Extract aligned VAD tensors
vad_text_full    = vad_raw['vad_text'].numpy()    # [13072, 3]
vad_image_full   = vad_raw['vad_image'].numpy()   # [13072, 3]
affective_meta_full = vad_raw['affective_meta']   # [13072, 128]

vad_text    = np.zeros((N, 3))
vad_image   = np.zeros((N, 3))
affective_meta = torch.zeros(N, affective_meta_full.shape[1])

valid_vad = 0
for i, idx in enumerate(vad_indices):
    if idx is not None and idx < len(vad_text_full):
        vad_text[i]  = vad_text_full[idx]
        vad_image[i] = vad_image_full[idx]
        affective_meta[i] = affective_meta_full[idx]
        valid_vad += 1

print(f"   ✅ VAD data aligned: {valid_vad}/{N} posts matched")

# Add mismatch magnitude to df
df['mismatch_magnitude'] = v_mismatch.norm(dim=1).numpy()

# Anomaly assignments
anomaly_df = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")
anomaly_df['post_id'] = anomaly_df['post_id'].astype(str)
df = df.merge(anomaly_df[['post_id','anomaly_score','anomaly_level',
                            'iso_forest_score','lof_score','ocsvm_score','elliptic_score']],
              on='post_id', how='left')

# GNN predictions
gnn_preds = pd.read_csv("gnn_results/gnn_post_predictions.csv")
gnn_preds['post_id'] = gnn_preds['post_id'].astype(str)
df = df.merge(gnn_preds[['post_id','fake_prob','predicted','split']], on='post_id', how='left')

print(f"   ✅ Master dataframe: {len(df)} posts, {len(df.columns)} columns")

# ============================================================================
# STEP 2: FIX 2 — Recompute method flags from anomaly_assignments.csv
#                  using 75th percentile on REAL posts only
# ============================================================================
print("\n[STEP 2] Recomputing method agreement flags (Fix 2)...")

real_posts = df[df['label_binary'] == 0].copy()

score_cols = {
    'iso_forest_flag':  'iso_forest_score',
    'lof_flag':         'lof_score',
    'ocsvm_flag':       'ocsvm_score',
    'elliptic_flag':    'elliptic_score',
}

thresholds = {}
for flag_col, score_col in score_cols.items():
    if score_col in real_posts.columns:
        thresh = real_posts[score_col].quantile(0.75)
        thresholds[flag_col] = thresh
        df[flag_col] = (df[score_col] > thresh).astype(int)
        print(f"   {flag_col}: threshold={thresh:.4f}, "
              f"flagged={df[flag_col].sum()} posts")

flag_cols = list(thresholds.keys())
df['n_methods_flagged'] = df[flag_cols].sum(axis=1).astype(int)

# Validate gradient
print(f"\n   Method agreement gradient (real-post thresholds):")
for n in range(5):
    subset = df[df['n_methods_flagged'] == n]
    if len(subset) > 0:
        fake_rate = subset['label_binary'].mean()
        print(f"   {n} methods: {fake_rate:.1%} fake (n={len(subset)})")

# ============================================================================
# STEP 3: FIX 3 — Load fusion weights with correct key names
# ============================================================================
print("\n[STEP 3] Extracting fusion weights (Fix 3 — correct key names)...")

FUSION_AVAILABLE = False
fusion_df = None

try:
    from rough_work import EmotionAwareFakeNewsDetector
    from torch.utils.data import Dataset, DataLoader

    # Load all required inputs — aligned to cluster order
    with open("Dataset/twitter/image_embeddings_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    image_emb_full = cache["image_embeddings"].float()  # may be 13072 or 10826

    text_col = 'post_text' if 'post_text' in df.columns else 'text'
    semantic_vectors = np.array(df["semantic_vector"].tolist())
    text_embeddings  = torch.tensor(semantic_vectors, dtype=torch.float32)

    meta_emb_full = torch.load("metadata_user_sequence_embeddings.pt")
    if meta_emb_full.dim() == 3:
        meta_emb_full = meta_emb_full.squeeze(1)

    # Align image and meta embeddings to cluster order using same VAD alignment
    image_embeddings = torch.zeros(N, image_emb_full.shape[1])
    meta_embeddings  = torch.zeros(N, meta_emb_full.shape[1])
    for i, idx in enumerate(vad_indices):
        if idx is not None:
            if idx < len(image_emb_full):
                image_embeddings[i] = image_emb_full[idx]
            if idx < len(meta_emb_full):
                meta_embeddings[i]  = meta_emb_full[idx]

    # Load checkpoint state
    state = torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=device)

    # Build model
    emotion_model = EmotionAwareFakeNewsDetector(
        d_text=128, d_image=1024, d_meta=128, d_common=256,
        vad_dim=3, meta_affective_dim=128, mismatch_dim=128,
        temporal_hidden=64, num_classes=1
    ).to(device)

    # ── Fix 3a: rebuild classifier with correct indices [0,3,6,9] ──
    def rebuild_seq(state_dict, prefix, indices):
        layers = []
        for i, idx in enumerate(indices):
            w = state_dict[f"{prefix}.{idx}.weight"]
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if i < len(indices) - 1:
                layers += [nn.ReLU(), nn.Dropout(0.3)]
        return nn.Sequential(*layers)

    emotion_model.classifier = rebuild_seq(state, "classifier", [0, 3, 6, 9]).to(device)

    # ── Fix 3b: rebuild gating_network with correct prefix 'fusion_layer.' ──
    gn_layers = []
    # indices: 0, 3, 5 (from checkpoint keys)
    w0 = state["fusion_layer.emotion_gate.gating_network.0.weight"]
    gn_layers.append(nn.Linear(w0.shape[1], w0.shape[0]))
    gn_layers.append(nn.ReLU())
    gn_layers.append(nn.Dropout(0.3))
    w3 = state["fusion_layer.emotion_gate.gating_network.3.weight"]
    gn_layers.append(nn.Linear(w3.shape[1], w3.shape[0]))
    gn_layers.append(nn.ReLU())
    w5 = state["fusion_layer.emotion_gate.gating_network.5.weight"]
    gn_layers.append(nn.Linear(w5.shape[1], w5.shape[0]))
    emotion_model.fusion_layer.emotion_gate.gating_network = nn.Sequential(*gn_layers).to(device)

    # ── Fix 3c: rebuild mismatch_encoder with correct prefix ──
    emotion_model.fusion_layer.emotion_gate.mismatch_generator.mismatch_encoder = \
        rebuild_seq(state,
                    "fusion_layer.emotion_gate.mismatch_generator.mismatch_encoder",
                    [0, 3, 6]).to(device)

    # ── Load state dict directly — no key renaming needed ──
    missing, unexpected = emotion_model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, \
        f"Load mismatch!\nMissing: {missing}\nUnexpected: {unexpected}"
    emotion_model.eval()
    print("   ✅ Emotion model loaded with correct keys")

    # Extract fusion weights
    class AlignedDataset(Dataset):
        def __init__(self, text, image, meta, vad_t, vad_i, aff_m):
            self.text = text
            self.image = image
            self.meta = meta
            self.vad_t = vad_t
            self.vad_i = vad_i
            self.aff_m = aff_m
        def __len__(self): return len(self.text)
        def __getitem__(self, i):
            return (self.text[i], self.image[i], self.meta[i],
                    self.vad_t[i], self.vad_i[i], self.aff_m[i])

    vad_text_t    = torch.tensor(vad_text,  dtype=torch.float32)
    vad_image_t   = torch.tensor(vad_image, dtype=torch.float32)

    dataset = AlignedDataset(text_embeddings, image_embeddings, meta_embeddings,
                              vad_text_t, vad_image_t, affective_meta)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)

    all_weights = []
    with torch.no_grad():
        for batch in loader:
            h_t, h_i, h_m, vt, vi, am = [b.to(device) for b in batch]
            _, intermediates = emotion_model(h_t, h_i, h_m,
                                             vad_text=vt, vad_image=vi,
                                             affective_meta=am)
            all_weights.append(intermediates['emotion_weights'].cpu())

    fusion_tensor = torch.cat(all_weights, dim=0)
    fusion_df = pd.DataFrame({
        'post_id':      df['post_id'].values,
        'label_binary': df['label_binary'].values,
        'label':        df['label'].values,
        'text_weight':  fusion_tensor[:, 0].numpy(),
        'image_weight': fusion_tensor[:, 1].numpy(),
        'meta_weight':  fusion_tensor[:, 2].numpy(),
    })
    fusion_df['dominant_modality'] = [
        'text' if t >= i and t >= m else 'image' if i >= m else 'meta'
        for t, i, m in zip(fusion_df['text_weight'],
                           fusion_df['image_weight'],
                           fusion_df['meta_weight'])
    ]
    fusion_df.to_csv(output_dir / "fusion_weights_all_posts.csv", index=False)
    print(f"   ✅ Fusion weights extracted for {len(fusion_df)} posts")
    FUSION_AVAILABLE = True

except Exception as e:
    print(f"   ⚠️  Fusion weight extraction failed: {e}")
    import traceback; traceback.print_exc()

# ============================================================================
# STEP 4: FIX 1 — VAD emotion analysis with correct alignment
# ============================================================================
print("\n[STEP 4] VAD emotion analysis (Fix 1 — aligned VAD)...")

fake_mask = df['label_binary'].values == 1
real_mask = df['label_binary'].values == 0

dims = ['valence', 'arousal', 'dominance']

print("   TEXT VAD — fake vs real:")
text_results = {}
for i, dim in enumerate(dims):
    f_vals = vad_text[fake_mask, i]
    r_vals = vad_text[real_mask, i]
    t_stat, p_val = ttest_ind(f_vals, r_vals)
    from numpy import sqrt
    d = (f_vals.mean() - r_vals.mean()) / sqrt((f_vals.std()**2 + r_vals.std()**2) / 2)
    text_results[dim] = {'fake_mean': f_vals.mean(), 'real_mean': r_vals.mean(),
                          't': t_stat, 'p': p_val, 'd': d}
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"   {dim:12s}: fake={f_vals.mean():.4f}, real={r_vals.mean():.4f}, "
          f"p={p_val:.6f} {sig}, d={d:.3f}")

print("\n   IMAGE VAD — fake vs real:")
image_results = {}
for i, dim in enumerate(dims):
    f_vals = vad_image[fake_mask, i]
    r_vals = vad_image[real_mask, i]
    t_stat, p_val = ttest_ind(f_vals, r_vals)
    d = (f_vals.mean() - r_vals.mean()) / sqrt((f_vals.std()**2 + r_vals.std()**2) / 2)
    image_results[dim] = {'fake_mean': f_vals.mean(), 'real_mean': r_vals.mean(),
                           't': t_stat, 'p': p_val, 'd': d}
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"   {dim:12s}: fake={f_vals.mean():.4f}, real={r_vals.mean():.4f}, "
          f"p={p_val:.6f} {sig}, d={d:.3f}")

print("\n   TEXT-IMAGE DISAGREEMENT per dimension — fake vs real:")
disagreement = np.abs(vad_text - vad_image)
disagree_results = {}
for i, dim in enumerate(dims):
    f_vals = disagreement[fake_mask, i]
    r_vals = disagreement[real_mask, i]
    t_stat, p_val = ttest_ind(f_vals, r_vals)
    d = (f_vals.mean() - r_vals.mean()) / sqrt((f_vals.std()**2 + r_vals.std()**2) / 2)
    disagree_results[dim] = {'fake_mean': f_vals.mean(), 'real_mean': r_vals.mean(),
                              't': t_stat, 'p': p_val, 'd': d}
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"   {dim:12s}: fake={f_vals.mean():.4f}, real={r_vals.mean():.4f}, "
          f"p={p_val:.6f} {sig}, d={d:.3f}")

# ============================================================================
# STEP 5: TEXT ATTRIBUTION
# ============================================================================
print("\n[STEP 5] Text attribution...")

text_col = 'post_text' if 'post_text' in df.columns else 'text'
texts  = df[text_col].fillna('').astype(str).tolist()
labels = df['label_binary'].values

vectorizer   = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

word_scores = {}
for i, word in enumerate(feature_names):
    word_tfidf = tfidf_matrix[:, i].toarray().flatten()
    corr, pval = spearmanr(word_tfidf, labels)
    if not np.isnan(corr):
        word_scores[word] = {'correlation': corr, 'abs_corr': abs(corr), 'pval': pval}

word_df = pd.DataFrame(word_scores).T.reset_index()
word_df.columns = ['word', 'correlation', 'abs_corr', 'pval']
word_df = word_df.sort_values('abs_corr', ascending=False)

top_fake_words = word_df[word_df['correlation'] > 0].head(15)
top_real_words = word_df[word_df['correlation'] < 0].head(15)

print(f"   Top fake words: {', '.join(top_fake_words.head(5)['word'].tolist())}")
print(f"   Top real words: {', '.join(top_real_words.head(5)['word'].tolist())}")

# ============================================================================
# STEP 6: GNN EMBEDDINGS
# ============================================================================
print("\n[STEP 6] Loading GNN embeddings...")

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

node_features_graph, edge_dict, node_mappings = load_heterogeneous_graph()
for ntype in node_features_graph:
    feat = node_features_graph[ntype].float()
    feat = torch.nan_to_num(feat, nan=0.0)
    node_features_graph[ntype] = F.normalize(feat, p=2, dim=1).to(device)
edge_dict_device = {k: (ei.to(device), ew.to(device).float()) for k, (ei, ew) in edge_dict.items()}

checkpoint = torch.load("checkpoints/best_het_gnn.pth", map_location=device, weights_only=False)
gnn_model = TemporalHeterogeneousGNN(
    node_dims=checkpoint['node_dims'], hidden_dim=256, num_layers=3,
    relation_types=checkpoint['relation_types'], num_classes=2
).to(device)
gnn_model.load_state_dict(checkpoint['model_state'])
gnn_model.eval()

timestamps_dict  = torch.load("heterogeneous_graph/node_timestamps.pt")
timestamps_device = {k: v.float().to(device) for k, v in timestamps_dict.items()}
current_time = float(timestamps_device['post'].max().item())

with torch.no_grad():
    outputs = gnn_model(node_features_graph, edge_dict_device,
                        timestamps=timestamps_device, current_time=current_time,
                        classify_edges=False)
    post_embeddings = outputs['embeddings']['post'].cpu().numpy()
    post_probs_gnn  = F.softmax(outputs['node_logits']['post'], dim=1)[:, 1].cpu().numpy()

post_labels = torch.load("heterogeneous_graph/post_labels.pt").numpy()
print(f"   ✅ GNN embeddings: {post_embeddings.shape}")

# ============================================================================
# STEP 7: CASE STUDIES
# ============================================================================
print("\n[STEP 7] Generating case studies...")

def get_top_words(text, ws, top_k=5):
    if not isinstance(text, str):
        return []
    words = [w.lower().strip('.,!?#@') for w in text.split()]
    scored = [(w, ws[w]['abs_corr']) for w in words if w in ws]
    return [w for w, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]

high_conf = df[
    (df['label_binary'] == 1) &
    (df['n_methods_flagged'] >= 3) &
    (df['mismatch_magnitude'].notna())
].sort_values('anomaly_score', ascending=False).head(10)

case_studies = []
for _, row in high_conf.iterrows():
    pid  = row['post_id']
    text = str(row.get(text_col, ''))
    top_words = get_top_words(text, word_scores)

    if FUSION_AVAILABLE:
        fw_row = fusion_df[fusion_df['post_id'] == pid]
        if len(fw_row) > 0:
            fw = fw_row.iloc[0]
            dominant = fw['dominant_modality']
            weights_str = f"text={fw['text_weight']:.2f} img={fw['image_weight']:.2f} meta={fw['meta_weight']:.2f}"
        else:
            dominant, weights_str = 'unknown', 'N/A'
    else:
        dominant, weights_str = 'unknown', 'N/A'

    gnn_row  = gnn_preds[gnn_preds['post_id'] == pid]
    gnn_prob = float(gnn_row.iloc[0]['fake_prob']) if len(gnn_row) > 0 else np.nan

    case_studies.append({
        'post_id':             pid,
        'true_label':          'fake',
        'anomaly_level':       row.get('anomaly_level', 'unknown'),
        'anomaly_score':       round(float(row['anomaly_score']), 4),
        'mismatch_magnitude':  round(float(row['mismatch_magnitude']), 4),
        'n_methods_flagged':   int(row['n_methods_flagged']),
        'gnn_fake_prob':       round(gnn_prob, 4) if not np.isnan(gnn_prob) else None,
        'dominant_modality':   dominant,
        'fusion_weights':      weights_str,
        'top_suspicious_words': ', '.join(top_words),
        'text_preview':        text[:150],
    })

case_df = pd.DataFrame(case_studies)
case_df.to_csv(output_dir / "case_studies.csv", index=False)

with open(output_dir / "case_studies_report.txt", 'w') as f:
    f.write("CASE STUDIES — DETECTED FAKE POSTS\n")
    f.write("=" * 80 + "\n\n")
    for i, row in case_df.iterrows():
        f.write(f"Case {i+1}: Post {row['post_id']}\n")
        f.write(f"  Text:               {row['text_preview']}\n")
        f.write(f"  Anomaly level:      {row['anomaly_level']}\n")
        f.write(f"  Anomaly score:      {row['anomaly_score']}\n")
        f.write(f"  Mismatch magnitude: {row['mismatch_magnitude']}\n")
        f.write(f"  Methods flagged:    {row['n_methods_flagged']}/4\n")
        f.write(f"  GNN fake prob:      {row['gnn_fake_prob']}\n")
        f.write(f"  Dominant modality:  {row['dominant_modality']}\n")
        f.write(f"  Fusion weights:     {row['fusion_weights']}\n")
        f.write(f"  Suspicious words:   {row['top_suspicious_words']}\n")
        f.write("\n" + "-" * 80 + "\n\n")

print(f"   ✅ {len(case_df)} case studies saved")

# ============================================================================
# STEP 8: VISUALISATIONS
# ============================================================================
print("\n[STEP 8] Generating visualisations...")

# ── Fig 1: VAD emotion analysis ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('VAD Emotion Analysis — Fake vs Real Posts', fontsize=15, fontweight='bold')

for i, dim in enumerate(dims):
    # Text VAD
    ax = axes[0, i]
    r = text_results[dim]
    ax.hist(vad_text[real_mask, i], bins=40, alpha=0.6,
            color='steelblue', label=f"Real (μ={r['real_mean']:.3f})", density=True)
    ax.hist(vad_text[fake_mask, i], bins=40, alpha=0.6,
            color='coral', label=f"Fake (μ={r['fake_mean']:.3f})", density=True)
    ax.axvline(r['real_mean'], color='steelblue', linestyle='--', linewidth=2)
    ax.axvline(r['fake_mean'], color='coral',     linestyle='--', linewidth=2)
    sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else "ns"
    ax.set_title(f"Text {dim.title()}\np={r['p']:.4f} {sig}, d={r['d']:.3f}",
                 fontweight='bold')
    ax.set_xlabel(f'{dim.title()} Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for i, dim in enumerate(dims):
    # Text-image disagreement
    ax = axes[1, i]
    r = disagree_results[dim]
    ax.hist(disagreement[real_mask, i], bins=40, alpha=0.6,
            color='steelblue', label=f"Real (μ={r['real_mean']:.3f})", density=True)
    ax.hist(disagreement[fake_mask, i], bins=40, alpha=0.6,
            color='coral', label=f"Fake (μ={r['fake_mean']:.3f})", density=True)
    ax.axvline(r['real_mean'], color='steelblue', linestyle='--', linewidth=2)
    ax.axvline(r['fake_mean'], color='coral',     linestyle='--', linewidth=2)
    sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else "ns"
    ax.set_title(f"Text-Image {dim.title()} Disagreement\np={r['p']:.4f} {sig}, d={r['d']:.3f}",
                 fontweight='bold')
    ax.set_xlabel(f'|text_{dim} - image_{dim}|')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "vad_emotion_analysis.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved vad_emotion_analysis.png")

# ── Fig 2: Arousal scatter — emotion space ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Emotion Space — Valence vs Arousal', fontsize=14, fontweight='bold')

sample_size = min(3000, N)
rng = np.random.default_rng(42)
sample_idx = rng.choice(N, sample_size, replace=False)
colors = np.array(['steelblue' if l == 0 else 'coral'
                   for l in df['label_binary'].values[sample_idx]])

ax = axes[0]
ax.scatter(vad_text[sample_idx, 0], vad_text[sample_idx, 1],
           c=colors, s=8, alpha=0.4, linewidths=0)
ax.set_xlabel('Valence'); ax.set_ylabel('Arousal')
ax.set_title('Text VAD Space')
real_p = mpatches.Patch(color='steelblue', label='Real')
fake_p = mpatches.Patch(color='coral',     label='Fake')
ax.legend(handles=[real_p, fake_p])
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.scatter(vad_image[sample_idx, 0], vad_image[sample_idx, 1],
           c=colors, s=8, alpha=0.4, linewidths=0)
ax.set_xlabel('Valence'); ax.set_ylabel('Arousal')
ax.set_title('Image VAD Space')
ax.legend(handles=[real_p, fake_p])
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(output_dir / "emotion_space_scatter.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved emotion_space_scatter.png")

# ── Fig 3: Mismatch by anomaly level ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Mismatch Magnitude Analysis', fontsize=14, fontweight='bold')

ax = axes[0]
fake_mm = df[df['label_binary'] == 1]['mismatch_magnitude'].dropna()
real_mm = df[df['label_binary'] == 0]['mismatch_magnitude'].dropna()
t, p = ttest_ind(fake_mm, real_mm)
ax.hist(real_mm, bins=40, alpha=0.6, color='steelblue',
        label=f'Real (μ={real_mm.mean():.4f})', density=True)
ax.hist(fake_mm, bins=40, alpha=0.6, color='coral',
        label=f'Fake (μ={fake_mm.mean():.4f})', density=True)
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns (p={:.3f})".format(p)
ax.set_title(f'Mismatch by Label\n{sig}')
ax.set_xlabel('||v_mismatch||'); ax.set_ylabel('Density')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
level_order = ['normal', 'low', 'medium', 'high', 'critical']
level_data  = [df[df['anomaly_level'] == l]['mismatch_magnitude'].dropna()
               for l in level_order]
level_means = [d.mean() for d in level_data]
level_ns    = [len(d) for d in level_data]
colors_l    = ['#95a5a6', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
bars = ax.bar(level_order, level_means, color=colors_l, edgecolor='black', alpha=0.85)
for bar, n in zip(bars, level_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'n={n}', ha='center', va='bottom', fontsize=8)
ax.set_title('Mismatch by Anomaly Level\n(Monotonic increase validates signal)')
ax.set_xlabel('Anomaly Level'); ax.set_ylabel('Mean Mismatch Magnitude')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "mismatch_analysis.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved mismatch_analysis.png")

# ── Fig 4: Fusion weights (if available) ────────────────────────────────────
if FUSION_AVAILABLE:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Emotion-Gated Modality Fusion Weights', fontsize=14, fontweight='bold')

    fake_fw = fusion_df[fusion_df['label_binary'] == 1]
    real_fw = fusion_df[fusion_df['label_binary'] == 0]

    for i, mod in enumerate(['text_weight', 'image_weight', 'meta_weight']):
        ax = axes[i]
        t, p = ttest_ind(fake_fw[mod], real_fw[mod])
        ax.hist(real_fw[mod], bins=30, alpha=0.65, color='steelblue',
                label=f'Real (μ={real_fw[mod].mean():.3f})', density=True)
        ax.hist(fake_fw[mod], bins=30, alpha=0.65, color='coral',
                label=f'Fake (μ={fake_fw[mod].mean():.3f})', density=True)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f'{mod.replace("_weight","").title()} Weight\np={p:.4f} {sig}')
        ax.set_xlabel('Weight'); ax.set_ylabel('Density')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fusion_weight_analysis.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved fusion_weight_analysis.png")

    # Dominant modality breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    mods = ['text', 'image', 'meta']
    dom_fake = fusion_df[fusion_df['label_binary'] == 1]['dominant_modality'].value_counts()
    dom_real = fusion_df[fusion_df['label_binary'] == 0]['dominant_modality'].value_counts()
    x = np.arange(3)
    ax.bar(x - 0.175, [dom_fake.get(m, 0) for m in mods], 0.35,
           label='Fake', color='coral', edgecolor='black', alpha=0.85)
    ax.bar(x + 0.175, [dom_real.get(m, 0) for m in mods], 0.35,
           label='Real', color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['Text', 'Image', 'Metadata'])
    ax.set_ylabel('Number of Posts')
    ax.set_title('Dominant Modality by Label')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "dominant_modality.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved dominant_modality.png")

# ── Fig 5: Text attribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Text Attribution — TF-IDF Correlation with Fake/Real', fontsize=14, fontweight='bold')

ax = axes[0]
top_f = word_df[word_df['correlation'] > 0].head(15)
ax.barh(top_f['word'], top_f['correlation'], color='coral', edgecolor='black', alpha=0.85)
ax.set_xlabel('Spearman r with Fake Label')
ax.set_title('Words → Fake Posts')
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)

ax = axes[1]
top_r = word_df[word_df['correlation'] < 0].head(15)
ax.barh(top_r['word'], top_r['correlation'].abs(), color='steelblue', edgecolor='black', alpha=0.85)
ax.set_xlabel('|Spearman r| with Real Label')
ax.set_title('Words → Real Posts')
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "text_attribution.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved text_attribution.png")

# ── Fig 6: Method agreement gradient ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
agreement_stats = df.groupby('n_methods_flagged').agg(
    fake_rate=('label_binary', 'mean'),
    n_posts=('label_binary', 'count')
).reset_index()

bar_colors = ['#95a5a6','#f39c12','#e67e22','#e74c3c','#8e44ad'][:len(agreement_stats)]
bars = ax.bar(agreement_stats['n_methods_flagged'],
              agreement_stats['fake_rate'] * 100,
              color=bar_colors, edgecolor='black', alpha=0.85, width=0.6)
ax.axhline(df['label_binary'].mean() * 100, color='navy',
           linestyle='--', linewidth=2,
           label=f'Baseline ({df["label_binary"].mean():.1%})')
for bar, row in zip(bars, agreement_stats.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'n={row.n_posts}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Number of Detection Methods in Agreement')
ax.set_ylabel('Fake News Rate (%)')
ax.set_title('Ensemble Agreement → Fake Rate Gradient\n(Computed from real-post thresholds)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_xticks(agreement_stats['n_methods_flagged'])
plt.tight_layout()
plt.savefig(output_dir / "method_agreement_gradient.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved method_agreement_gradient.png")

# ── Fig 7: GNN UMAP ──────────────────────────────────────────────────────────
try:
    from umap import UMAP
    print("   Running UMAP on GNN embeddings...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    emb_2d  = reducer.fit_transform(post_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('GNN Post Embeddings — UMAP Projection', fontsize=14, fontweight='bold')

    c_label = np.array(['steelblue' if l == 0 else 'coral' for l in post_labels])
    axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=c_label, s=5, alpha=0.4, linewidths=0)
    axes[0].set_title('Coloured by True Label')
    axes[0].legend(handles=[mpatches.Patch(color='steelblue', label='Real'),
                             mpatches.Patch(color='coral', label='Fake')])
    axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
    axes[0].grid(True, alpha=0.2)

    sc = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=post_probs_gnn,
                         cmap='RdYlBu_r', s=5, alpha=0.4, linewidths=0, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], label='GNN Fake Probability')
    axes[1].set_title('Coloured by GNN Fake Probability')
    axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "gnn_embedding_umap.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved gnn_embedding_umap.png")
except ImportError:
    print("   ⚠️  umap-learn not installed — pip3 install umap-learn")

# ============================================================================
# STEP 9: SUMMARY
# ============================================================================
print("\n[STEP 9] Writing summary...")

sig_findings = []
for dim in dims:
    r = text_results[dim]
    if r['p'] < 0.05:
        sig_findings.append(f"Text {dim}: p={r['p']:.4f}, d={r['d']:.3f}")
for dim in dims:
    r = disagree_results[dim]
    if r['p'] < 0.05:
        sig_findings.append(f"Text-Image {dim} disagreement: p={r['p']:.4f}, d={r['d']:.3f}")

summary = f"""
EXPLAINABILITY SUMMARY — FIXED VERSION
{'='*60}

SIGNIFICANT VAD FINDINGS:
{chr(10).join('  ' + f for f in sig_findings) if sig_findings else '  None significant — check VAD alignment'}

METHOD AGREEMENT GRADIENT (recomputed from real-post thresholds):
"""
for _, row in agreement_stats.iterrows():
    summary += f"  {int(row['n_methods_flagged'])} methods: {row['fake_rate']:.1%} fake (n={int(row['n_posts'])})\n"

summary += f"""
FUSION WEIGHTS: {'Extracted ✅' if FUSION_AVAILABLE else 'Failed ⚠️'}

GNN: val F1={checkpoint['val_f1']:.4f}, val AUC={checkpoint['val_auc']:.4f}

TEXT ATTRIBUTION:
  Top fake words: {', '.join(top_fake_words.head(5)['word'].tolist())}
  Top real words: {', '.join(top_real_words.head(5)['word'].tolist())}

OUTPUT FILES:
  vad_emotion_analysis.png      — VAD distributions fake vs real
  emotion_space_scatter.png     — valence-arousal scatter
  mismatch_analysis.png         — mismatch by label and anomaly level
  fusion_weight_analysis.png    {'✅' if FUSION_AVAILABLE else '⚠️ skipped'}
  dominant_modality.png         {'✅' if FUSION_AVAILABLE else '⚠️ skipped'}
  text_attribution.png          — TF-IDF word correlations
  method_agreement_gradient.png — ensemble validation
  gnn_embedding_umap.png        — GNN representation space
  case_studies_report.txt       — 10 qualitative examples
"""

print(summary)
with open(output_dir / "explainability_summary.txt", 'w') as f:
    f.write(summary)

print("=" * 80)
print("✅ EXPLAINABILITY COMPLETE")
print("=" * 80)