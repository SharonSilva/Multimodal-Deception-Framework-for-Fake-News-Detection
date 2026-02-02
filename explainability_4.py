# explainability_descriptive_user_v2.py
# ============================================================
# DESCRIPTIVE HUMAN-FRIENDLY EXPLAINABILITY WITH FACTS
# ============================================================

import torch
import numpy as np
import pandas as pd
from datetime import datetime

print("="*80)
print("LOADING DATA FOR DESCRIPTIVE EXPLANATIONS WITH FACTS")
print("="*80)

# -----------------------------
# Load prepared embeddings
# -----------------------------
cluster_data = torch.load("prepared_clustering_data.pt", weights_only=False)

z_aug = cluster_data["z_out"]
v_mismatch = cluster_data["v_mismatch"]
user_ids = cluster_data["user_ids"]
timestamps = cluster_data["timestamps"]
post_ids = cluster_data["post_ids"]

# -----------------------------
# Load anomaly detection models
# This file is locally generated and trusted, so weights_only=False is fine.
# -----------------------------
anomaly_data = torch.load(
    "anomaly_detection_results/anomaly_models.pt",
    map_location="cpu",
    weights_only=False
)

# -----------------------------
# Load ensemble anomaly info
# -----------------------------
X_reduced = anomaly_data["X_reduced"]

# Compute individual model scores
iso_scores = anomaly_data["models"]["isolation_forest"].decision_function(X_reduced)
lof_scores = anomaly_data["models"]["local_outlier_factor"].decision_function(X_reduced) if "local_outlier_factor" in anomaly_data["models"] else anomaly_data["models"]["ocsvm"].decision_function(X_reduced)
ocsvm_scores = anomaly_data["models"]["ocsvm"].score_samples(X_reduced)
elliptic_scores = anomaly_data["models"]["elliptic"].score_samples(X_reduced)

# Normalize each score to [0, 1] where 1 = most anomalous
# IsolationForest: lower decision_function = more anomalous
iso_norm = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
# LOF: lower decision_function = more anomalous
lof_norm = 1 - (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
# OCSVM: lower score_samples = more anomalous
ocsvm_norm = 1 - (ocsvm_scores - ocsvm_scores.min()) / (ocsvm_scores.max() - ocsvm_scores.min() + 1e-8)
# Elliptic: lower score_samples = more anomalous
elliptic_norm = 1 - (elliptic_scores - elliptic_scores.min()) / (elliptic_scores.max() - elliptic_scores.min() + 1e-8)

# Ensemble score = average of normalized scores
ensemble_score = (iso_norm + lof_norm + ocsvm_norm + elliptic_norm) / 4.0

# Assign anomaly levels based on ensemble score
anomaly_levels = np.where(
    ensemble_score > 0.75, "critical",
    np.where(ensemble_score > 0.5, "high",
    np.where(ensemble_score > 0.25, "moderate", "normal"))
)

print(f"📊 Ensemble anomaly score stats:")
print(f"   Min: {ensemble_score.min():.4f}, Max: {ensemble_score.max():.4f}, "
      f"Mean: {ensemble_score.mean():.4f}")
print(f"   Critical: {(anomaly_levels == 'critical').sum()}, "
      f"High: {(anomaly_levels == 'high').sum()}, "
      f"Moderate: {(anomaly_levels == 'moderate').sum()}, "
      f"Normal: {(anomaly_levels == 'normal').sum()}")

# -----------------------------
# Load campaign detection results
# -----------------------------
campaign_assignments = pd.read_csv("campaign_detection_results/campaign_assignments.csv")
campaign_stats = pd.read_csv("campaign_detection_results/campaign_statistics.csv")

# Build per-post campaign lookup: post_id → campaign_id
campaign_map = dict(zip(
    campaign_assignments['post_id'].astype(str),
    campaign_assignments['campaign_id']
))

# Build per-campaign coordination score lookup: campaign_id → coordination_score
coordination_map = dict(zip(
    campaign_stats['campaign_id'],
    campaign_stats['coordination_score']
))

# Build per-campaign size lookup: campaign_id → n_posts
campaign_size_map = dict(zip(
    campaign_stats['campaign_id'],
    campaign_stats['n_posts']
))

# Compute user repeat rate per campaign:
# For each (user, campaign) pair, how many posts does that user have in that campaign?
user_campaign_counts = (
    campaign_assignments
    .groupby(['user_id', 'campaign_id'])['post_id']
    .count()
    .reset_index()
    .rename(columns={'post_id': 'user_posts_in_campaign'})
)
# Merge back so each row knows how many times that user posted in that campaign
campaign_assignments = campaign_assignments.merge(
    user_campaign_counts, on=['user_id', 'campaign_id'], how='left'
)
# Build lookup: (user_id, post_id) → user_posts_in_campaign
user_repeat_map = {}
for _, row in campaign_assignments.iterrows():
    user_repeat_map[str(row['post_id'])] = int(row['user_posts_in_campaign'])

print(f"📊 Campaign data loaded:")
print(f"   Posts with campaign assignments: {len(campaign_assignments)}")
print(f"   Total campaigns: {len(campaign_stats)}")
print(f"   Campaigns with coordination_score > 1.0: {(campaign_stats['coordination_score'] > 1.0).sum()}")
# Raw v_mismatch norms are unbounded (you're seeing 1.44, 1.80 etc.)
# Normalize by dividing by the max so thresholds make sense
contradiction_score_raw = torch.norm(v_mismatch, dim=1).numpy()
contradiction_score = contradiction_score_raw / contradiction_score_raw.max()

print(f"📊 Contradiction score stats (normalized):")
print(f"   Min: {contradiction_score.min():.4f}, Max: {contradiction_score.max():.4f}, "
      f"Mean: {contradiction_score.mean():.4f}, Median: {np.median(contradiction_score):.4f}")

# -------------------------------------------
# 2️⃣ EMOTIONAL INTENSITY & GAPS
# -------------------------------------------
vad_vectors = z_aug[:, :3]  # first 3 dims as VAD
emotional_intensity = np.linalg.norm(vad_vectors, axis=1)
emotional_gaps = np.abs(np.diff(vad_vectors, axis=0)).sum(axis=1)

# -------------------------------------------
# 3️⃣ NARRATIVE SIMILARITY
# -------------------------------------------
z_norm = z_aug / (z_aug.norm(dim=1, keepdim=True) + 1e-8)
cos_sim_matrix = torch.matmul(z_norm, z_norm.T).numpy()
narrative_similarity = cos_sim_matrix.mean(axis=1)

# -------------------------------------------
# 4️⃣ TEMPORAL REUSE (Corrected)
# -------------------------------------------
timestamps_np = np.array(timestamps)
temporal_deltas = np.diff(timestamps_np, prepend=timestamps_np[0])

# Convert delta seconds to minutes for readability
temporal_minutes = temporal_deltas / 60.0

# Classify temporal reuse into slow/moderate/rapid
temporal_reuse_labels = []
for delta_min in temporal_minutes:
    if delta_min < 10:  # less than 10 minutes → rapid
        temporal_reuse_labels.append("rapid")
    elif delta_min < 60:  # 10-60 minutes → moderate
        temporal_reuse_labels.append("moderate")
    else:
        temporal_reuse_labels.append("slow")

temporal_reuse_values = 1 / (1 + temporal_deltas)  # keep numeric value if needed

# -------------------------------------------
# 5️⃣ GENERATE DESCRIPTIVE EXPLANATIONS
# -------------------------------------------
descriptive_explanations = []

# Pre-compute all percentiles once before the loop
contradiction_p90 = np.percentile(contradiction_score, 90)
contradiction_p75 = np.percentile(contradiction_score, 75)
contradiction_p25 = np.percentile(contradiction_score, 25)
emotion_p75 = np.percentile(emotional_intensity, 75)
emotion_p25 = np.percentile(emotional_intensity, 25)
gap_p75 = np.percentile(emotional_gaps, 75)
narrative_p75 = np.percentile(narrative_similarity, 75)
narrative_p25 = np.percentile(narrative_similarity, 25)

print(f"📊 Contradiction thresholds → p25:{contradiction_p25:.3f}, p75:{contradiction_p75:.3f}, p90:{contradiction_p90:.3f}")
print(f"📊 Emotion intensity thresholds → p25:{emotion_p25:.3f}, p75:{emotion_p75:.3f}")

for i in range(len(post_ids)):
    post_id = post_ids[i]
    user_id = user_ids[i]
    post_time = datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d %H:%M:%S")
    
    # Contradiction / Conflicting content
    score = contradiction_score[i]

    if score > contradiction_p90 or score > 0.95:
        contradiction_text = (
            f"🚨 Contradiction Score: {score:.2f} (critical). "
            "Strong conflict between this post and other content — high deception risk."
        )
    elif score > contradiction_p75 or score > 0.90:
        contradiction_text = (
            f"⚠️ Contradiction Score: {score:.2f} (high). "
            "This post contradicts other content, which may confuse readers."
        )
    elif score > contradiction_p25:
        contradiction_text = (
            f"🟡 Contradiction Score: {score:.2f} (moderate). "
            "Some inconsistency with surrounding content."
        )
    else:
        contradiction_text = (
            f"✅ Contradiction Score: {score:.2f} (low). "
            "This post aligns with previous content."
        )
    
    # Emotional tone
    if emotional_intensity[i] > emotion_p75:
        emotion_text = (
            f"💥 Emotional Intensity: {emotional_intensity[i]:.2f} (strong). "
            "Likely intended to grab attention or provoke reactions."
        )
    elif emotional_intensity[i] < emotion_p25:
        emotion_text = (
            f"😐 Emotional Intensity: {emotional_intensity[i]:.2f} (calm). "
            "Tone is neutral."
        )
    else:
        emotion_text = (
            f"🙂 Emotional Intensity: {emotional_intensity[i]:.2f} (moderate)."
        )
    
    # Emotional gap
    gap_text = ""
    if i > 0 and emotional_gaps[i-1] > gap_p75:
        gap_text = (
            f"⚡ Emotional Gap: {emotional_gaps[i-1]:.2f} (high). "
            "Noticeable shift in sentiment from previous post."
        )
    
    # Narrative similarity
    if narrative_similarity[i] > narrative_p75:
        narrative_text = (
            f"🔁 Narrative Similarity: {narrative_similarity[i]:.2f} (high). "
            "Post repeats ideas from other posts, suggesting coordinated narrative."
        )
    elif narrative_similarity[i] < narrative_p25:
        narrative_text = (
            f"🆕 Narrative Similarity: {narrative_similarity[i]:.2f} (low). "
            "Introduces new ideas."
        )
    else:
        narrative_text = (
            f"🔄 Narrative Similarity: {narrative_similarity[i]:.2f} (moderate). "
            "Partially aligns with existing narratives."
        )
    
    # Temporal spread
    temporal_text = (
        f"⏱️ Temporal Reuse: {temporal_reuse_values[i]:.4f} ({temporal_reuse_labels[i]}). "
        "Content spreading speed after similar posts."
    )
    
    # Campaign membership
    post_id_str = str(post_id)
    campaign_id = campaign_map.get(post_id_str, None)

    if campaign_id is not None:
        coord_score = coordination_map.get(campaign_id, 0)
        camp_size = campaign_size_map.get(campaign_id, 0)
        user_repeats = user_repeat_map.get(post_id_str, 1)

        if coord_score > 1.0:
            campaign_emoji = "🚨"
            campaign_label = "high-coordination"
        elif coord_score > 0.5:
            campaign_emoji = "⚠️"
            campaign_label = "moderate-coordination"
        else:
            campaign_emoji = "🟡"
            campaign_label = "low-coordination"

        campaign_text = (
            f"{campaign_emoji} Campaign: #{campaign_id} ({campaign_label}). "
            f"{camp_size} posts in this campaign. "
            f"This user posted {user_repeats}x in it"
            f"{' — repeated posting suggests coordination.' if user_repeats >= 3 else '.'}"
        )
    else:
        campaign_text = (
            f"✅ Campaign: None. This post is not part of any detected coordinated campaign."
        )
    if anomaly_levels[i] == "critical":
        anomaly_emoji = "🚨"
    elif anomaly_levels[i] == "high":
        anomaly_emoji = "⚠️"
    elif anomaly_levels[i] == "moderate":
        anomaly_emoji = "🟡"
    else:
        anomaly_emoji = "✅"

    anomaly_text = (
        f"{anomaly_emoji} Anomaly Score: {ensemble_score[i]:.2f} ({anomaly_levels[i]}). "
        f"Method scores → Iso:{iso_norm[i]:.2f}, LOF:{lof_norm[i]:.2f}, "
        f"OCSVM:{ocsvm_norm[i]:.2f}, Elliptic:{elliptic_norm[i]:.2f}"
    )
    
    # Combine into a paragraph
    explanation = (
        f"Post {post_id} by {user_id} at {post_time}:\n"
        f"- {contradiction_text}\n"
        f"- {emotion_text}\n"
        f"{('- ' + gap_text + chr(10)) if gap_text else ''}"
        f"- {narrative_text}\n"
        f"- {temporal_text}\n"
        f"- {campaign_text}\n"
        f"- {anomaly_text}\n"
    )
    
    descriptive_explanations.append(explanation)

# -------------------------------------------
# SAVE DESCRIPTIVE EXPLANATIONS
# -------------------------------------------
df_descriptive = pd.DataFrame({
    "post_id": post_ids,
    "user_id": user_ids,
    "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
    "campaign_id": [campaign_map.get(str(pid), -1) for pid in post_ids],
    "coordination_score": [coordination_map.get(campaign_map.get(str(pid)), 0.0) for pid in post_ids],
    "user_posts_in_campaign": [user_repeat_map.get(str(pid), 0) for pid in post_ids],
    "explanation": descriptive_explanations
})

df_descriptive.to_csv("explainability_descriptive_user_v2.csv", index=False)
print("="*80)
print("✅ Descriptive explanations with facts saved as explainability_descriptive_user_v2.csv")

# Print first 5
print("\n🔥 FIRST 5 DESCRIPTIVE EXPLANATIONS 🔥\n")
for explanation in descriptive_explanations[:5]:
    print(explanation)
    print("-"*80)