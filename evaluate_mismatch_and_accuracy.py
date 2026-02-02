#!/usr/bin/env python3
"""
evaluate_mismatch_and_accuracy.py

Robusted and fixed version:
 - handles messy label strings
 - aligns dataframe rows with cached embeddings when rows are dropped
 - safer semantic_vector parsing
 - mismatch diagnostics, cross-verifier evaluation, ablation
"""

import os
import pickle
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config / paths
# -------------------------
CACHE_PATH = "Dataset/twitter/image_embeddings_cache.pkl"
DF_PATH = "Dataset/twitter/df_train_translated.csv"
METADATA_EMB_PT = "metadata_user_sequence_embeddings.pt"   # saved by your earlier script
METADATA_DENSE_PT = "metadata_dense_embeddings.pt"
LABEL_COL = "label"   # change if your label column is named differently
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# Helper: load df and embeddings
# -------------------------
print("Loading dataframe...")
df = pd.read_csv(DF_PATH)
print("Dataframe loaded: rows =", len(df))

if LABEL_COL not in df.columns:
    raise RuntimeError(f"Label column '{LABEL_COL}' not found in dataframe. Add a label column or set LABEL_COL correctly.")

# Normalize label strings
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()

# Common mappings (expand if you have other variants)
label_map = {
    'fake': 0, 'f': 0, '0': 0, 'false': 0, 'not_real': 0,
    'real': 1, 'r': 1, '1': 1, 'true': 1
}

# Attempt mapping; any unmapped will become NaN
df['_mapped_label'] = df[LABEL_COL].map(label_map)

# Print unique original labels to help debugging if some are unmapped
unique_labels = pd.Series(df[LABEL_COL].unique()).tolist()
print("Unique label strings (sample):", unique_labels[:50])

# Drop rows with unmapped labels (or handle differently if you want)
num_unmapped = df['_mapped_label'].isna().sum()
if num_unmapped > 0:
    print(f"Warning: {num_unmapped} rows had unmapped labels and will be dropped. Example unmapped values:")
    print(df.loc[df['_mapped_label'].isna(), LABEL_COL].drop_duplicates().head(10).tolist())

valid_mask = ~df['_mapped_label'].isna()
df = df.loc[valid_mask].reset_index(drop=True)
df[LABEL_COL] = df['_mapped_label'].astype(int)
df.drop(columns=['_mapped_label'], inplace=True)
print("After dropping unmapped labels: rows =", len(df))

print("Loading cached image/text embeddings...")
if not os.path.exists(CACHE_PATH):
    raise RuntimeError(f"Cache not found at {CACHE_PATH}. Run embedding script first.")

with open(CACHE_PATH, "rb") as f:
    cached = pickle.load(f)

image_embeddings = cached.get('image_embeddings')
text_embeddings = cached.get('text_embeddings')

if image_embeddings is None or text_embeddings is None:
    raise RuntimeError("Cache file does not contain 'image_embeddings' and 'text_embeddings' keys.")

# Convert torch -> numpy if necessary
if isinstance(image_embeddings, torch.Tensor):
    image_embeddings = image_embeddings.cpu().numpy()
if isinstance(text_embeddings, torch.Tensor):
    text_embeddings = text_embeddings.cpu().numpy()

# Align embeddings to filtered df.
# The embeddings were created from the original dataframe order. We dropped rows from df, so we need to apply the same mask.
# The evaluate script only dropped rows due to unmapped labels; we must recreate that mask relative to original file.
# To do that, reload original labels quickly and compute mapping mask.
orig_df = pd.read_csv(DF_PATH)
orig_labels_norm = orig_df[LABEL_COL].astype(str).str.lower().str.strip()
orig_mapped = orig_labels_norm.map(label_map)
orig_valid_mask = ~orig_mapped.isna()

# Sanity: number of valid rows in orig_valid_mask should match len(image_embeddings)
if orig_valid_mask.sum() != image_embeddings.shape[0]:
    # If mismatch, try alternative: if embeddings length equals original df length, assume embeddings include unmapped rows.
    print("Warning: couldn't infer exact original mask. Attempting alignment heuristics...")
    if len(orig_df) == image_embeddings.shape[0]:
        print("Heuristic: embeddings appear aligned to original df (no drop) - using current df indices to select subset.")
        # Build index mapping: select indices of orig rows that remain in df by matching a unique id column if available.
        # Try 'post_id' or 'id' or use index if unique and preserved.
        unique_col = None
        for candidate in ['post_id', 'id', 'image_id']:
            if candidate in orig_df.columns and candidate in df.columns:
                unique_col = candidate
                break
        if unique_col is not None:
            print(f"Using unique column '{unique_col}' to align embeddings.")
            # create mapping from orig index order to new df
            orig_index_map = orig_df[unique_col].astype(str).tolist()
            sel_ids = df[unique_col].astype(str).tolist()
            idx_map = [orig_index_map.index(sid) for sid in sel_ids]
            image_embeddings = image_embeddings[idx_map]
            text_embeddings = text_embeddings[idx_map]
        else:
            # fallback: if lengths equal after drop, just truncate/pad (not ideal)
            print("No unique id to align. Falling back to using first N rows of embeddings where N = len(df).")
            image_embeddings = image_embeddings[:len(df)]
            text_embeddings = text_embeddings[:len(df)]
    else:
        # If shapes differ but original valid mask sum equals embeddings length, use that
        if orig_valid_mask.sum() == image_embeddings.shape[0]:
            idxs = np.where(orig_valid_mask)[0].tolist()
            # select those indices but also filter to only those that remain in current df (we dropped some)
            # To reduce complexity: assume we only dropped those rows (labels unmapped) -> select in same order
            image_embeddings = image_embeddings
            text_embeddings = text_embeddings
        else:
            raise RuntimeError("Unable to align embeddings with filtered dataframe. Please ensure you created embeddings from the same CSV.")

# Now ensure shapes are consistent
n_samples = image_embeddings.shape[0]
if text_embeddings.shape[0] != n_samples:
    raise RuntimeError("Text/image embeddings length mismatch: %d vs %d" % (text_embeddings.shape[0], n_samples))

# If df length differs from embeddings length, try to align by index — simplest approach: if df length <= embeddings length, assume embeddings are prefix
if len(df) != n_samples:
    print(f"Warning: dataframe rows ({len(df)}) != embeddings ({n_samples}). Attempting to align by truncation/pad.")
    if len(df) < n_samples:
        image_embeddings = image_embeddings[:len(df)]
        text_embeddings = text_embeddings[:len(df)]
        n_samples = len(df)
    else:
        # pad embeddings with zeros (unlikely but safe)
        pad_n = len(df) - n_samples
        image_embeddings = np.vstack([image_embeddings, np.zeros((pad_n, image_embeddings.shape[1]), dtype=np.float32)])
        text_embeddings = np.vstack([text_embeddings, np.zeros((pad_n, text_embeddings.shape[1]), dtype=np.float32)])
        n_samples = len(df)

# metadata: try to load saved user sequence embeddings (torch.np)
if os.path.exists(METADATA_EMB_PT):
    print("Loading metadata user sequence embeddings...")
    metadata_user_seq = torch.load(METADATA_EMB_PT)
    if isinstance(metadata_user_seq, torch.Tensor):
        arr = metadata_user_seq.cpu().numpy()
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        metadata_embeddings = arr
    else:
        metadata_embeddings = np.array(metadata_user_seq)
    # align if lengths mismatch
    if metadata_embeddings.shape[0] != n_samples:
        print("Metadata emb length mismatch -> truncating/padding to match embeddings length.")
        if metadata_embeddings.shape[0] >= n_samples:
            metadata_embeddings = metadata_embeddings[:n_samples]
        else:
            pad = np.zeros((n_samples - metadata_embeddings.shape[0], metadata_embeddings.shape[1]), dtype=np.float32)
            metadata_embeddings = np.vstack([metadata_embeddings, pad])
elif os.path.exists(METADATA_DENSE_PT):
    print("Loading metadata dense embeddings...")
    dense = torch.load(METADATA_DENSE_PT)
    metadata_embeddings = dense.cpu().numpy()
    if metadata_embeddings.shape[0] != n_samples:
        metadata_embeddings = metadata_embeddings[:n_samples]
else:
    # Fallback: create trivial metadata features from numeric counts in df
    print("No metadata emb file found. Creating simple metadata features from df columns as fallback.")
    numeric_cols = [c for c in ['hashtags_count','user_mentions_count','urls_count','emojis_count','num_posts_user'] if c in df.columns]
    if len(numeric_cols) == 0:
        metadata_embeddings = np.zeros((n_samples, 16), dtype=np.float32)
    else:
        arr = df[numeric_cols].fillna(0).astype(float).values
        metadata_embeddings = StandardScaler().fit_transform(arr)
        if metadata_embeddings.shape[1] < 128:
            pad = np.zeros((n_samples, 128 - metadata_embeddings.shape[1]), dtype=np.float32)
            metadata_embeddings = np.hstack([metadata_embeddings, pad])
        else:
            metadata_embeddings = metadata_embeddings[:, :128]

# text semantic vector: either df['semantic_vector'] (list of lists) or use text_embeddings
if 'semantic_vector' in df.columns:
    print("Parsing 'semantic_vector' column from dataframe...")
    sem_list = []
    for v in df['semantic_vector'].values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            sem_list.append(np.zeros(text_embeddings.shape[1], dtype=np.float32))
            continue
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
            except Exception:
                # fallback if CSV saved as string with brackets but not strict python
                try:
                    parsed = np.fromstring(v.strip('[]'), sep=',')
                except Exception:
                    parsed = np.zeros(text_embeddings.shape[1], dtype=np.float32)
            sem_list.append(np.array(parsed, dtype=np.float32))
        elif isinstance(v, (list, tuple, np.ndarray)):
            sem_list.append(np.array(v, dtype=np.float32))
        else:
            sem_list.append(np.zeros(text_embeddings.shape[1], dtype=np.float32))
    # pad/truncate each to consistent dim
    max_dim = max([arr.shape[0] for arr in sem_list])
    if max_dim != text_embeddings.shape[1]:
        # prefer text_embeddings dim
        target_dim = text_embeddings.shape[1]
        sem_array = np.zeros((n_samples, target_dim), dtype=np.float32)
        for i, arr in enumerate(sem_list):
            arr = arr.flatten()
            if arr.shape[0] >= target_dim:
                sem_array[i] = arr[:target_dim]
            else:
                sem_array[i, :arr.shape[0]] = arr
        semantic_vectors = sem_array
    else:
        semantic_vectors = np.stack([arr if arr.shape[0]==max_dim else np.pad(arr, (0,max_dim-arr.shape[0])) for arr in sem_list])
else:
    semantic_vectors = text_embeddings.copy()

# Ensure all arrays have same first dimension n_samples
assert image_embeddings.shape[0] == semantic_vectors.shape[0] == metadata_embeddings.shape[0] == len(df), \
    f"Mismatch shapes: image {image_embeddings.shape[0]}, text {semantic_vectors.shape[0]}, meta {metadata_embeddings.shape[0]}, df {len(df)}"

y = df[LABEL_COL].astype(int).values
print(f"Loaded: n_samples={n_samples}, image_emb.shape={image_embeddings.shape}, text_emb.shape={semantic_vectors.shape}, meta_emb.shape={metadata_embeddings.shape}")

# -------------------------
# Define compact models (reuse simplified classes)
# -------------------------
class Step1_InputProjection(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.proj_text = nn.Sequential(nn.Linear(d_text, d_common), nn.ReLU())
        self.proj_image = nn.Sequential(nn.Linear(d_image, d_common), nn.ReLU())
        self.proj_meta = nn.Sequential(nn.Linear(d_meta, d_common), nn.ReLU())
        self.norm_text = nn.LayerNorm(d_common)
        self.norm_image = nn.LayerNorm(d_common)
        self.norm_meta = nn.LayerNorm(d_common)
    def forward(self, h_text, h_image, h_meta):
        zt = self.norm_text(self.proj_text(h_text))
        zi = self.norm_image(self.proj_image(h_image))
        zm = self.norm_meta(self.proj_meta(h_meta))
        return zt, zi, zm

class Step3_MismatchVector(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(d_common, d_common),
            nn.ReLU(),
            nn.Linear(d_common, d_common)
        )
        self.norm = nn.LayerNorm(d_common)
    def forward(self, zt, zi):
        diff = zt - zi
        v = self.mismatch_encoder(diff)
        return self.norm(v)

class Step4_AdaptiveGating(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_common*4, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_common)
    def forward(self, zt, zi, zm, v_mismatch):
        concat = torch.cat([zt, zi, zm, v_mismatch], dim=1)
        alpha = self.gate(concat)
        z_fused = alpha[:,0:1]*zt + alpha[:,1:2]*zi + alpha[:,2:3]*zm
        return self.norm(z_fused), alpha

class Step5_FinalFusion(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))
    def forward(self, z_fused, v_mismatch):
        return torch.cat([z_fused, self.gamma * v_mismatch], dim=1)

class AdaptiveMultimodalFakeNewsDetector(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.step1 = Step1_InputProjection(d_text, d_image, d_meta, d_common)
        self.mismatch = Step3_MismatchVector(d_common)
        self.gating = Step4_AdaptiveGating(d_common)
        self.final = Step5_FinalFusion(d_common)
        self.classifier = nn.Sequential(
            nn.Linear(2*d_common, d_common),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_common, 1)
        )
    def forward(self, h_text, h_image, h_meta, return_intermediates=False):
        zt, zi, zm = self.step1(h_text, h_image, h_meta)
        v_mismatch = self.mismatch(zt, zi)
        z_fused, alpha = self.gating(zt, zi, zm, v_mismatch)
        z_out = self.final(z_fused, v_mismatch)
        logits = self.classifier(z_out)
        if return_intermediates:
            return logits, {'zt':zt, 'zi':zi, 'zm':zm, 'v_mismatch':v_mismatch, 'alpha':alpha}
        return logits

# CrossVerifier (simple)
class InnerFusionModule(nn.Module):
    def __init__(self, text_dim=128, image_dim=1024, fused_dim=1024):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.cross_gate = nn.Linear(fused_dim*2, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
    def forward(self, text_emb, image_emb):
        t = torch.relu(self.text_proj(text_emb))
        v = torch.relu(self.image_proj(image_emb))
        fused = torch.relu(self.cross_gate(torch.cat([t,v], dim=1)))
        return self.norm(fused)

class CrossVerifier(nn.Module):
    def __init__(self, fused_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(fused_dim, fused_dim//2)
        self.fc2 = nn.Linear(fused_dim//2, 1)
    def forward(self, fused_emb):
        x = torch.relu(self.fc1(fused_emb))
        return torch.sigmoid(self.fc2(x))

# -------------------------
# Prepare dataset
# -------------------------
X_text = semantic_vectors.astype(np.float32)
X_image = image_embeddings.astype(np.float32)
X_meta = metadata_embeddings.astype(np.float32)

n_samples = X_text.shape[0]
y = y.astype(np.int64)

train_idx, test_idx = train_test_split(np.arange(n_samples), test_size=0.2, random_state=SEED, stratify=y)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED, stratify=y[train_idx])

def make_loader(idxs, batch_size=BATCH_SIZE, shuffle=True):
    t_text = torch.tensor(X_text[idxs]).to(torch.float32)
    t_img  = torch.tensor(X_image[idxs]).to(torch.float32)
    t_meta = torch.tensor(X_meta[idxs]).to(torch.float32)
    t_lab  = torch.tensor(y[idxs]).to(torch.float32)
    ds = TensorDataset(t_text, t_img, t_meta, t_lab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(train_idx, shuffle=True)
val_loader = make_loader(val_idx, shuffle=False)
test_loader = make_loader(test_idx, shuffle=False)

# -------------------------
# Train adaptive model (quick)
# -------------------------
adaptive = AdaptiveMultimodalFakeNewsDetector(d_text=X_text.shape[1], d_image=X_image.shape[1],
                                             d_meta=X_meta.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(adaptive.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for t_text, t_img, t_meta, t_lab in loader:
        t_text = t_text.to(DEVICE); t_img = t_img.to(DEVICE); t_meta = t_meta.to(DEVICE); t_lab = t_lab.to(DEVICE)
        optimizer.zero_grad()
        logits = model(t_text, t_img, t_meta)
        loss = criterion(logits.squeeze(), t_lab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * t_text.size(0)
    return total_loss / (len(loader.dataset))

def eval_model(model, loader):
    model.eval()
    all_preds = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for t_text, t_img, t_meta, t_lab in loader:
            t_text = t_text.to(DEVICE); t_img = t_img.to(DEVICE); t_meta = t_meta.to(DEVICE)
            logits = model(t_text, t_img, t_meta)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_scores.extend(probs.tolist())
            all_labels.extend(t_lab.numpy().astype(int).tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = float('nan')
    return {'accuracy':acc, 'f1':f1, 'precision':prec, 'recall':rec, 'auc':auc}

print("Training adaptive model for a few epochs (increase epochs for real use)...")
for ep in range(5):
    tr_loss = train_one_epoch(adaptive, train_loader)
    val_metrics = eval_model(adaptive, val_loader)
    print(f"Epoch {ep+1}: train_loss={tr_loss:.4f} | val_f1={val_metrics['f1']:.4f}, val_auc={val_metrics['auc']:.4f}")

test_metrics = eval_model(adaptive, test_loader)
print("Adaptive model test metrics:", test_metrics)

# -------------------------
# CrossVerifier evaluation
# -------------------------
print("\nEvaluating CrossVerifier on balanced pair set (pos = (text_i,image_i), neg = (text_i,image_j))...")
n = n_samples
indices = np.arange(n)
pairs_neg = np.random.permutation(indices)

text_pairs = np.concatenate([X_text, X_text], axis=0)
img_pairs  = np.concatenate([X_image, X_image[pairs_neg]], axis=0)
pair_labels = np.concatenate([np.ones(n), np.zeros(n)], axis=0)

T_text = torch.tensor(text_pairs, dtype=torch.float32).to(DEVICE)
T_img  = torch.tensor(img_pairs, dtype=torch.float32).to(DEVICE)

cross_fuser = InnerFusionModule(text_dim=T_text.shape[1], image_dim=T_img.shape[1]).to(DEVICE)
cross_verifier = CrossVerifier(fused_dim=1024).to(DEVICE)

# quick train on subset
max_train_pairs = min(2000, 2*n)
indices_train = np.random.choice(2*n, size=max_train_pairs, replace=False)
X_t_train = T_text[indices_train]
X_i_train = T_img[indices_train]
y_train_cv = torch.tensor(pair_labels[indices_train], dtype=torch.float32).to(DEVICE)

optimizer_cv = torch.optim.Adam(list(cross_fuser.parameters()) + list(cross_verifier.parameters()), lr=1e-4)
bce = nn.BCELoss()

for epoch in range(3):
    perm = np.random.permutation(len(indices_train))
    total_loss = 0.0
    batch_size_cv = 128
    cross_fuser.train(); cross_verifier.train()
    for bstart in range(0, len(perm), batch_size_cv):
        bidx = perm[bstart:bstart+batch_size_cv]
        tbat = X_t_train[bidx]; ibat = X_i_train[bidx]; lab = y_train_cv[bidx]
        optimizer_cv.zero_grad()
        fused = cross_fuser(tbat, ibat)
        preds = cross_verifier(fused).squeeze()
        loss = bce(preds, lab)
        loss.backward(); optimizer_cv.step()
        total_loss += loss.item() * len(bidx)
    print(f"CrossVerifier Train Epoch {epoch+1}: avg_loss={total_loss/len(perm):.4f}")

with torch.no_grad():
    fused_all = cross_fuser(T_text, T_img)
    scores = cross_verifier(fused_all).squeeze().cpu().numpy()
    try:
        auc_cv = roc_auc_score(pair_labels, scores)
    except Exception:
        auc_cv = float('nan')
    pred_bin = (scores > 0.5).astype(int)
    acc_cv = accuracy_score(pair_labels, pred_bin)
    print("CrossVerifier metrics on pair set: AUC=%.4f, Acc=%.4f" % (auc_cv, acc_cv))

# -------------------------
# Mismatch vector diagnostics
# -------------------------
print("\nComputing mismatch vector magnitudes & correlations (full dataset)...")
adaptive.eval()
all_v = []
all_alpha = []
all_logits = []
with torch.no_grad():
    for bstart in range(0, n, BATCH_SIZE):
        bend = min(bstart+BATCH_SIZE, n)
        t_text = torch.tensor(X_text[bstart:bend], dtype=torch.float32).to(DEVICE)
        t_img  = torch.tensor(X_image[bstart:bend], dtype=torch.float32).to(DEVICE)
        t_meta = torch.tensor(X_meta[bstart:bend], dtype=torch.float32).to(DEVICE)
        logits, inter = adaptive(t_text, t_img, t_meta, return_intermediates=True)
        v = inter['v_mismatch'].cpu().numpy()
        a = inter['alpha'].cpu().numpy()
        s = torch.sigmoid(logits.squeeze()).cpu().numpy()
        all_v.append(v); all_alpha.append(a); all_logits.append(s)
all_v = np.vstack(all_v)
all_alpha = np.vstack(all_alpha)
all_logits = np.concatenate(all_logits)

mismatch_magnitudes = np.linalg.norm(all_v, axis=1)
corr_label, p = spearmanr(mismatch_magnitudes, y)
corr_score, p2 = spearmanr(mismatch_magnitudes, all_logits)
print(f"Spearman corr(mismatch_mag, label) = {corr_label:.4f} (p={p:.4g})")
print(f"Spearman corr(mismatch_mag, adaptive_score) = {corr_score:.4f} (p={p2:.4g})")

# hist plot
plt.figure(figsize=(6,4))
plt.hist(mismatch_magnitudes[y==0], bins=50, alpha=0.6, label='real (0)')
plt.hist(mismatch_magnitudes[y==1], bins=50, alpha=0.6, label='fake (1)')
plt.legend()
plt.title("Mismatch magnitude distribution by label")
plt.xlabel("||v_mismatch||")
plt.savefig("mismatch_magnitude_hist.png", dpi=150)
print("Saved mismatch magnitude histogram: mismatch_magnitude_hist.png")

# t-SNE (subsample)
subsample = min(2000, len(all_v))
idxs = np.random.choice(len(all_v), subsample, replace=False)
tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, max(5, subsample//10)))
X2 = tsne.fit_transform(all_v[idxs])
plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=y[idxs], cmap='coolwarm', s=6, alpha=0.8)
plt.title("t-SNE of v_mismatch (subsample)")
plt.savefig("mismatch_tsne.png", dpi=150)
print("Saved t-SNE plot: mismatch_tsne.png")

# -------------------------
# Gate stats & ablation
# -------------------------
print("\nGate statistics (mean/std across dataset):")
print("alpha mean:", all_alpha.mean(axis=0))
print("alpha std: ", all_alpha.std(axis=0))

# Ablation: zero-out mismatch vector and evaluate
adaptive_abl = AdaptiveMultimodalFakeNewsDetector(d_text=X_text.shape[1], d_image=X_image.shape[1],
                                                 d_meta=X_meta.shape[1]).to(DEVICE)
adaptive_abl.load_state_dict(adaptive.state_dict())

def eval_with_mismatch_zeroed(model, loader):
    model.eval()
    all_preds = []; all_scores = []; all_labels = []
    with torch.no_grad():
        for t_text, t_img, t_meta, t_lab in loader:
            t_text = t_text.to(DEVICE); t_img = t_img.to(DEVICE); t_meta = t_meta.to(DEVICE)
            zt, zi, zm = model.step1(t_text, t_img, t_meta)
            v = model.mismatch(zt, zi)
            v_zero = torch.zeros_like(v)
            zf, a = model.gating(zt, zi, zm, v_zero)
            zout = model.final(zf, v_zero)
            logits = model.classifier(zout)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.tolist()); all_scores.extend(probs.tolist()); all_labels.extend(t_lab.numpy().astype(int).tolist())
    return {'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_scores)}

abl_metrics = eval_with_mismatch_zeroed(adaptive_abl, test_loader)
print("Ablation (mismatch zeroed) test metrics:", abl_metrics)
print("Delta (original - ablated):")
print({k: test_metrics[k] - abl_metrics[k] for k in test_metrics if k in abl_metrics})

# -------------------------
# Save artifacts + final prints
# -------------------------
print("\nFINAL SUMMARY")
print("Adaptive model test:", test_metrics)
print("CrossVerifier (pair dataset): AUC,Acc:", (auc_cv, acc_cv))
print("Mismatch correlation with labels:", corr_label)
print("Mismatch correlation with adaptive score:", corr_score)
print("Gate mean/std:", all_alpha.mean(axis=0), all_alpha.std(axis=0))

np.savez("evaluation_results.npz",
         adaptive_test_metrics=test_metrics,
         crossverifier_auc=auc_cv,
         crossverifier_acc=acc_cv,
         mismatch_magnitude=mismatch_magnitudes,
         gate_mean=all_alpha.mean(axis=0),
         gate_std=all_alpha.std(axis=0))

print("Saved evaluation_results.npz")
print("Done.")
