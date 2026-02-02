"""
prepare_clustering_data.py
================================
Prepares embeddings for campaign-level clustering & graph construction.
Uses ONLY the trained EmotionAwareFakeNewsDetector (which already does fusion).
Outputs:
 - z_aug (semantic + mismatch + affect dynamics)
 - v_mismatch
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import numpy as np

from tqdm import tqdm
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 80)
print("PREPARING DATA FOR CLUSTERING PIPELINE (CLEAN VERSION)")
print("=" * 80)

# ============================================================================
# STEP 1: Load dataframe
# ============================================================================

print("\n[1/6] Loading dataframe...")
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")

df = df.drop_duplicates(subset="post_id", keep="first").reset_index(drop=True)

if df["timestamp"].isnull().any():
    df["timestamp"] = df["timestamp"].fillna(df["timestamp"].median())

# Convert Twitter timestamp string → datetime
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce"
)

# Fill failed parses safely
if df["timestamp"].isnull().any():
    df["timestamp"] = df["timestamp"].fillna(df["timestamp"].median())

# Convert to Unix seconds
df["timestamp_unix"] = df["timestamp"].view("int64") // 10**9

print(f"✅ Posts after cleaning: {len(df)}")

# ============================================================================
# STEP 2: Load embeddings (NO RESIZING!)
# ============================================================================

print("\n[2/6] Loading embeddings...")

# ---- Text embeddings (128)
text_embeddings = torch.tensor(
    np.array(df["semantic_vector"].tolist()),
    dtype=torch.float32
)
print("✅ Text:", text_embeddings.shape)

# ---- Image embeddings (1024)
with open("Dataset/twitter/image_embeddings_cache.pkl", "rb") as f:
    cache = pickle.load(f)
image_embeddings = cache["image_embeddings"]

# Align length
N = len(df)
if len(image_embeddings) > N:
    image_embeddings = image_embeddings[:N]
elif len(image_embeddings) < N:
    pad = torch.zeros(N - len(image_embeddings), image_embeddings.shape[1])
    image_embeddings = torch.cat([image_embeddings, pad], dim=0)

print("✅ Image:", image_embeddings.shape)

# ---- Metadata embeddings (128)
metadata_embeddings = torch.load("metadata_user_sequence_embeddings.pt")
if metadata_embeddings.dim() == 3:
    metadata_embeddings = metadata_embeddings.squeeze(1)

if len(metadata_embeddings) > N:
    metadata_embeddings = metadata_embeddings[:N]
elif len(metadata_embeddings) < N:
    pad = torch.zeros(N - len(metadata_embeddings), metadata_embeddings.shape[1])
    metadata_embeddings = torch.cat([metadata_embeddings, pad], dim=0)

print("✅ Meta:", metadata_embeddings.shape)

# ---- VAD
vad_data = torch.load("Dataset/twitter/prepared_vad_data.pt")

for k in ["vad_text", "vad_image", "affective_meta"]:
    if len(vad_data[k]) > N:
        vad_data[k] = vad_data[k][:N]
    elif len(vad_data[k]) < N:
        pad = torch.zeros(N - len(vad_data[k]), vad_data[k].shape[1])
        vad_data[k] = torch.cat([vad_data[k], pad], dim=0)

print("✅ VAD text:", vad_data["vad_text"].shape)
print("✅ VAD image:", vad_data["vad_image"].shape)
print("✅ Affective meta:", vad_data["affective_meta"].shape)

# ============================================================================
# STEP 3: Dataset
# ============================================================================

class ClusteringDataset(Dataset):
    def __init__(self, df, text_emb, image_emb, meta_emb, vad_data):
        self.df = df
        self.text = text_emb
        self.image = image_emb
        self.meta = meta_emb
        self.vad = vad_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": self.text[idx],
            "image": self.image[idx],
            "meta": self.meta[idx],
            "vad_text": self.vad["vad_text"][idx],
            "vad_image": self.vad["vad_image"][idx],
            "affective_meta": self.vad["affective_meta"][idx],
            "user_id": row["username"],
            "timestamp": row["timestamp_unix"],
            "post_id": row["post_id"],
            "label": row["label"],
        }

dataset = ClusteringDataset(df, text_embeddings, image_embeddings, metadata_embeddings, vad_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"✅ Dataset ready: {len(dataset)} samples")

# ============================================================================
# STEP 4: Load TRAINED emotion-aware model
# ============================================================================

print("\n[4/6] Loading Emotion-Aware Model...")

from rough_work import EmotionAwareFakeNewsDetector

emotion_model = EmotionAwareFakeNewsDetector(
    d_text=128,
    d_image=1024,
    d_meta=128,
    d_common=256,                  # 🔴 MUST be 256
    vad_dim=3,
    meta_affective_dim=128,
    mismatch_dim=128,
    temporal_hidden=64,
    num_classes=1
).to(device)

# 🔴 LOAD TRAINED WEIGHTS
ckpt_path = "checkpoints/best_emotion_aware_detector.pth"   # <-- change if needed
state = torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=device)

# Remap keys: fusion.*  --> fusion_layer.*
new_state = {}
for k, v in state.items():
    if k.startswith("fusion."):
        new_k = k.replace("fusion.", "fusion_layer.")
        new_state[new_k] = v
    else:
        # skip classifier
        if not k.startswith("classifier."):
            new_state[k] = v

missing, unexpected = emotion_model.load_state_dict(new_state, strict=False)

print("Loaded remapped fusion weights.")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

emotion_model.eval()

print("✅ Emotion model loaded")

# ============================================================================
# STEP 5: Extract embeddings
# ============================================================================

print("\n[5/6] Extracting embeddings...")

all_z = []
all_v_mismatch = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        h_text = batch['text'].to(device)
        h_image = batch['image'].to(device)
        h_meta = batch['meta'].to(device)
        vad_text = batch['vad_text'].to(device)
        vad_image = batch['vad_image'].to(device)
        affective_meta = batch['affective_meta'].to(device)

        # ----- Step2: Emotion-aware model -----
        logits, intermediates = emotion_model(
            h_text, h_image, h_meta,
            affective_meta=affective_meta,
            vad_text=vad_text,
            vad_image=vad_image
        )
        
        z_aug = intermediates['z_aug']              # <-- real embedding
        v_mismatch = intermediates['v_mismatch']   # <-- mismatch vector

        all_z.append(z_aug.cpu())
        all_v_mismatch.append(v_mismatch.cpu())

z_tensor = torch.cat(all_z, dim=0)
v_mismatch_tensor = torch.cat(all_v_mismatch, dim=0)

print("z_aug:", z_tensor.shape)
print("v_mismatch:", v_mismatch_tensor.shape)

print("Sanity check:")
print("  z std:", z_tensor.std().item())
print("  v_mismatch std:", v_mismatch_tensor.std().item())

# ============================================================================
# STEP 6: Save
# ============================================================================

prepared = {
    "z_out": z_tensor[:, :128],  # Take first 128 dims to match anomaly script
    "v_mismatch": v_mismatch_tensor,
    "user_ids": df["username"].tolist(),
    "timestamps": df["timestamp_unix"].values,
    "post_ids": df["post_id"].tolist(),
    # optional extra fields
    "labels": df["label"].values,
    "meta": {
        "n_posts": len(df),
        "n_users": df["username"].nunique(),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
    }
}

torch.save(prepared, "prepared_clustering_data.pt")

print("\n✅ Saved: prepared_clustering_data.pt")
print("=" * 80)
print("✅ DATA PREPARATION COMPLETE")

