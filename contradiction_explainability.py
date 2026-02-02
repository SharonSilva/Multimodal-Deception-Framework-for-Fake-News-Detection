"""
EXPLAINABILITY LAYER WITH FUSION WEIGHTS AND CONTRADICTION SCORES
==================================================================
Enhanced explainability that includes:
- Modality fusion weights
- Contradiction scores (text-image mismatch)
- Suspicion detection patterns
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import networkx as nx
from datetime import datetime
import warnings
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings('ignore')

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph
from rough_work import EmotionAwareFakeNewsDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("EXPLAINABILITY LAYER WITH FUSION WEIGHTS & CONTRADICTION SCORES")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA WITH CONTRADICTION SCORES
# ============================================================================
print("\n[STEP 1] Loading data with contradiction scores...")

# Load the preprocessed dataframe that has contradiction scores
df_preprocessed = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
print(f"✅ Loaded preprocessed data with {len(df_preprocessed)} posts")
print(f"✅ Columns: {df_preprocessed.columns.tolist()}")

# Verify contradiction scores exist
if 'contradiction_score' in df_preprocessed.columns:
    print(f"✅ Found contradiction scores!")
    print(f"   Mean: {df_preprocessed['contradiction_score'].mean():.4f}")
    print(f"   Std: {df_preprocessed['contradiction_score'].std():.4f}")
    print(f"   Min: {df_preprocessed['contradiction_score'].min():.4f}")
    print(f"   Max: {df_preprocessed['contradiction_score'].max():.4f}")
else:
    print("⚠️ WARNING: No contradiction_score column found!")
    # Create dummy scores if missing
    df_preprocessed['contradiction_score'] = 0.0

# ============================================================================
# STEP 2: EXTRACT FUSION WEIGHTS FROM TRAINED MODEL
# ============================================================================
print("\n[STEP 2] Extracting fusion weights from trained model...")

# Load full dataframe with all features
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(int)

# Load embeddings
print("Loading embeddings...")
text_embeddings = torch.tensor(np.array(df["semantic_vector"].tolist()), dtype=torch.float32)

with open("Dataset/twitter/image_embeddings_cache.pkl", "rb") as f:
    cache = pickle.load(f)
image_embeddings = cache["image_embeddings"]

metadata_embeddings = torch.load("metadata_user_sequence_embeddings.pt")
if metadata_embeddings.dim() == 3:
    metadata_embeddings = metadata_embeddings.squeeze(1)

vad_data = torch.load("Dataset/twitter/prepared_vad_data.pt")

# Align sizes
N = len(df)
for emb in [image_embeddings, metadata_embeddings]:
    if len(emb) > N:
        emb = emb[:N]
    elif len(emb) < N:
        pad = torch.zeros(N - len(emb), emb.shape[1])
        emb = torch.cat([emb, pad], dim=0)

for k in ["vad_text", "vad_image", "affective_meta"]:
    if len(vad_data[k]) > N:
        vad_data[k] = vad_data[k][:N]
    elif len(vad_data[k]) < N:
        pad = torch.zeros(N - len(vad_data[k]), vad_data[k].shape[1])
        vad_data[k] = torch.cat([vad_data[k], pad], dim=0)

# Load trained model
print("Loading trained emotion-aware model...")
emotion_model = EmotionAwareFakeNewsDetector(
    d_text=128,
    d_image=1024,
    d_meta=128,
    d_common=256,
    vad_dim=3,
    meta_affective_dim=128,
    mismatch_dim=128,
    temporal_hidden=64,
    num_classes=1
).to(device)

state = torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=device)

# Remap keys
new_state = {}
for k, v in state.items():
    if k.startswith("fusion."):
        new_k = k.replace("fusion.", "fusion_layer.")
        new_state[new_k] = v
    else:
        if not k.startswith("classifier."):
            new_state[k] = v

emotion_model.load_state_dict(new_state, strict=False)
emotion_model.eval()

# Create dataset
class FusionWeightDataset(Dataset):
    def __init__(self, df, text_emb, image_emb, meta_emb, vad_data):
        self.df = df
        self.text = text_emb
        self.image = image_emb
        self.meta = meta_emb
        self.vad = vad_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "image": self.image[idx],
            "meta": self.meta[idx],
            "vad_text": self.vad["vad_text"][idx],
            "vad_image": self.vad["vad_image"][idx],
            "affective_meta": self.vad["affective_meta"][idx],
            "post_id": self.df.iloc[idx]["post_id"],
        }

dataset = FusionWeightDataset(df, text_embeddings, image_embeddings, metadata_embeddings, vad_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract fusion weights
print("Extracting fusion weights for all posts...")
all_fusion_weights = []
all_post_ids = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing batches"):
        h_text = batch['text'].to(device)
        h_image = batch['image'].to(device)
        h_meta = batch['meta'].to(device)
        vad_text = batch['vad_text'].to(device)
        vad_image = batch['vad_image'].to(device)
        affective_meta = batch['affective_meta'].to(device)

        logits, intermediates = emotion_model(
            h_text, h_image, h_meta,
            affective_meta=affective_meta,
            vad_text=vad_text,
            vad_image=vad_image
        )
        
        # Extract emotion_weights (fusion weights)
        emotion_weights = intermediates['emotion_weights']  # Shape: [batch, 3]
        
        all_fusion_weights.append(emotion_weights.cpu())
        all_post_ids.extend(batch['post_id'].tolist())

# Concatenate all weights
fusion_weights_tensor = torch.cat(all_fusion_weights, dim=0)  # Shape: [N, 3]

# Create fusion weights dataframe
fusion_df = pd.DataFrame({
    'post_id': all_post_ids,
    'text_weight': fusion_weights_tensor[:, 0].numpy(),
    'image_weight': fusion_weights_tensor[:, 1].numpy(),
    'meta_weight': fusion_weights_tensor[:, 2].numpy()
})

# Add dominant modality
modality_names = ['text', 'image', 'meta']
fusion_df['dominant_modality'] = [
    modality_names[idx] for idx in fusion_weights_tensor.argmax(dim=1).numpy()
]

print(f"✅ Extracted fusion weights for {len(fusion_df)} posts")

# ============================================================================
# STEP 3: MERGE ALL DATA TOGETHER
# ============================================================================
print("\n[STEP 3] Merging contradiction scores, fusion weights, and detections...")

# Load detection results
detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
high_conf = pd.read_csv("suspicious_detection_results/high_confidence_suspicious.csv")

# Merge everything together
# First merge fusion weights
detections = detections.merge(fusion_df, on='post_id', how='left')
high_conf = high_conf.merge(fusion_df, on='post_id', how='left')

# Then merge with df to get all features
merged = detections.merge(df, on='post_id', how='left')

# Handle contradiction scores - check if already in merged or need to add
if 'contradiction_score' not in merged.columns:
    if 'contradiction_score' in df_preprocessed.columns:
        contradiction_scores = df_preprocessed[['post_id', 'contradiction_score']].copy()
        merged = merged.merge(contradiction_scores, on='post_id', how='left')
        high_conf = high_conf.merge(contradiction_scores, on='post_id', how='left')
        print(f"✅ Merged contradiction scores")
    else:
        merged['contradiction_score'] = 0.0
        high_conf['contradiction_score'] = 0.0
        print(f"⚠️ No contradiction scores found, using zeros")
else:
    print(f"✅ Contradiction scores already present")

# Clean up duplicate columns if they exist
if 'contradiction_score_x' in merged.columns and 'contradiction_score_y' in merged.columns:
    print(f"⚠️ Duplicate contradiction_score columns found, cleaning up...")
    # Use _y (from df_preprocessed) if available, otherwise _x (from df)
    merged['contradiction_score'] = merged['contradiction_score_y'].fillna(merged['contradiction_score_x'])
    merged = merged.drop(columns=['contradiction_score_x', 'contradiction_score_y'])
    
    high_conf['contradiction_score'] = high_conf['contradiction_score_y'].fillna(high_conf['contradiction_score_x'])
    high_conf = high_conf.drop(columns=['contradiction_score_x', 'contradiction_score_y'])
    print(f"✅ Cleaned up duplicate columns")

print(f"✅ Merged {len(detections)} detections")
print(f"✅ Final columns: {merged.columns.tolist()}")

# ============================================================================
# STEP 4: TEXT ATTRIBUTION
# ============================================================================
print("\n[STEP 4] Computing Text Attribution Scores...")

def compute_text_attribution(text_series, suspicion_scores):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.stats import spearmanr
    
    texts = text_series.fillna('').astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    word_scores = {}
    for i, word in enumerate(feature_names):
        word_tfidf = tfidf_matrix[:, i].toarray().flatten()
        corr, _ = spearmanr(word_tfidf, suspicion_scores)
        if not np.isnan(corr):
            word_scores[word] = abs(corr)
    
    return word_scores, vectorizer

text_col = 'post_text' if 'post_text' in merged.columns else 'text'
word_scores, vectorizer = compute_text_attribution(
    merged[text_col], 
    merged['suspicion_score']
)

top_phrases = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:30]

print(f"\n📊 Top Suspicious Phrases:")
for phrase, score in top_phrases[:10]:
    print(f"   '{phrase}': {score:.4f}")

# ============================================================================
# STEP 5: ANALYZE FUSION WEIGHTS & CONTRADICTION SCORES
# ============================================================================
print("\n[STEP 5] Analyzing fusion weights and contradiction scores...")

# Analyze by detection category
suspicious_fusion = merged[merged['is_suspicious'] == True][['text_weight', 'image_weight', 'meta_weight', 'contradiction_score']]
normal_fusion = merged[merged['is_suspicious'] == False][['text_weight', 'image_weight', 'meta_weight', 'contradiction_score']]

print(f"\n📊 Average Fusion Weights - Suspicious Posts:")
print(f"   Text: {suspicious_fusion['text_weight'].mean():.4f}")
print(f"   Image: {suspicious_fusion['image_weight'].mean():.4f}")
print(f"   Meta: {suspicious_fusion['meta_weight'].mean():.4f}")
print(f"   Contradiction Score: {suspicious_fusion['contradiction_score'].mean():.4f}")

print(f"\n📊 Average Fusion Weights - Normal Posts:")
print(f"   Text: {normal_fusion['text_weight'].mean():.4f}")
print(f"   Image: {normal_fusion['image_weight'].mean():.4f}")
print(f"   Meta: {normal_fusion['meta_weight'].mean():.4f}")
print(f"   Contradiction Score: {normal_fusion['contradiction_score'].mean():.4f}")

# Correlation between contradiction score and suspicion
from scipy.stats import spearmanr
corr_contradiction_suspicion, p_value = spearmanr(
    merged['contradiction_score'].fillna(0), 
    merged['suspicion_score']
)
print(f"\n📊 Correlation: Contradiction Score ↔ Suspicion Score")
print(f"   Spearman r: {corr_contradiction_suspicion:.4f} (p={p_value:.4e})")

# ============================================================================
# STEP 6: GENERATE ENHANCED EXPLANATIONS
# ============================================================================
print("\n[STEP 6] Generating enhanced explanations...")

def generate_explanation_with_fusion_and_contradiction(post_row, word_scores, user_risk_scores=None):
    """Generate explanation including fusion weights AND contradiction scores"""
    explanation = {
        'post_id': post_row['post_id'],
        'suspicion_score': post_row['suspicion_score'],
        'contradiction_score': float(post_row.get('contradiction_score', 0)),
        'detection_methods': [],
        'text_highlights': [],
        'fusion_weights': {},
        'dominant_modality': post_row.get('dominant_modality', 'unknown'),
        'user_risk': None,
        'overall_summary': ''
    }
    
    # Detection methods
    if post_row.get('iso_forest_flag', 0):
        explanation['detection_methods'].append("Anomalous embedding pattern")
    if post_row.get('dbscan_outlier', 0):
        explanation['detection_methods'].append("Behavioral outlier")
    if post_row.get('high_distance', 0):
        explanation['detection_methods'].append("Isolated from normal content")
    if post_row.get('in_deception_cluster', 0):
        explanation['detection_methods'].append("Member of deception cluster")
    
    # Fusion weights
    explanation['fusion_weights'] = {
        'text': float(post_row.get('text_weight', 0)),
        'image': float(post_row.get('image_weight', 0)),
        'meta': float(post_row.get('meta_weight', 0))
    }
    
    # Text highlights
    text = str(post_row.get('post_text', post_row.get('text', '')))
    words = text.lower().split()
    highlighted_words = []
    for word in words:
        if word in word_scores and word_scores[word] > 0.3:
            highlighted_words.append((word, word_scores[word]))
    explanation['text_highlights'] = sorted(highlighted_words, key=lambda x: x[1], reverse=True)[:5]
    
    # Build summary
    num_methods = len(explanation['detection_methods'])
    suspicion = explanation['suspicion_score']
    contradiction = explanation['contradiction_score']
    
    if num_methods >= 3 and suspicion > 0.75:
        confidence = "Very High"
    elif num_methods >= 2 and suspicion > 0.5:
        confidence = "High"
    elif num_methods >= 1 and suspicion > 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    dominant_mod = explanation['dominant_modality']
    dominant_weight = explanation['fusion_weights'].get(dominant_mod, 0)
    
    explanation['overall_summary'] = (
        f"This post has a {confidence.lower()} confidence suspicion score of {suspicion:.2f}. "
        f"It was flagged by {num_methods} detection method(s). "
    )
    
    # Add contradiction information
    if contradiction > 0.5:
        explanation['overall_summary'] += (
            f"⚠️ HIGH TEXT-IMAGE CONTRADICTION detected (score: {contradiction:.2f}). "
            f"The textual content and visual content are mismatched, which is a strong indicator of deception. "
        )
    elif contradiction > 0.3:
        explanation['overall_summary'] += (
            f"Moderate text-image contradiction (score: {contradiction:.2f}). "
        )
    
    # Add modality information
    explanation['overall_summary'] += (
        f"The '{dominant_mod}' modality contributed most ({dominant_weight:.2%}) to this detection. "
    )
    
    if explanation['text_highlights'] and dominant_mod == 'text':
        top_word = explanation['text_highlights'][0][0]
        explanation['overall_summary'] += f"Key suspicious phrase: '{top_word}'. "
    elif dominant_mod == 'image':
        if contradiction > 0.5:
            explanation['overall_summary'] += "Visual content contradicts text and shows suspicious patterns. "
        else:
            explanation['overall_summary'] += "Visual content shows suspicious patterns. "
    elif dominant_mod == 'meta':
        explanation['overall_summary'] += "User behavior metadata indicates anomalous patterns. "
    
    return explanation

# Generate explanations for top suspicious posts
top_suspicious = high_conf.nlargest(20, 'suspicion_score')
explanations = []

for idx, row in top_suspicious.iterrows():
    exp = generate_explanation_with_fusion_and_contradiction(row, word_scores)
    explanations.append(exp)

# Save enhanced explanations
explanations_df = pd.DataFrame([
    {
        'post_id': exp['post_id'],
        'suspicion_score': exp['suspicion_score'],
        'contradiction_score': exp['contradiction_score'],
        'num_detection_methods': len(exp['detection_methods']),
        'detection_methods': '; '.join(exp['detection_methods']),
        'dominant_modality': exp['dominant_modality'],
        'text_weight': exp['fusion_weights']['text'],
        'image_weight': exp['fusion_weights']['image'],
        'meta_weight': exp['fusion_weights']['meta'],
        'top_suspicious_words': ', '.join([w for w, s in exp['text_highlights']]),
        'explanation_summary': exp['overall_summary']
    }
    for exp in explanations
])

output_dir = Path("explainability_results")
output_dir.mkdir(exist_ok=True)

explanations_df.to_csv(output_dir / 'post_explanations_with_fusion_and_contradiction.csv', index=False)
print(f"✅ Saved enhanced explanations for {len(explanations)} posts")

# ============================================================================
# STEP 7: VISUALIZATIONS WITH CONTRADICTION SCORES
# ============================================================================
print("\n[STEP 7] Creating visualizations with fusion weights and contradiction scores...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 7.1: Fusion weights distribution
ax = axes[0, 0]
fusion_weights_data = merged[['text_weight', 'image_weight', 'meta_weight']].values
ax.boxplot(fusion_weights_data, labels=['Text', 'Image', 'Meta'])
ax.set_ylabel('Weight Value')
ax.set_title('Fusion Weight Distribution', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 7.2: Contradiction Score Distribution
ax = axes[0, 1]
ax.hist(merged['contradiction_score'], bins=30, color='crimson', edgecolor='black', alpha=0.7)
ax.axvline(merged['contradiction_score'].median(), color='blue', linestyle='--', 
          label=f'Median: {merged["contradiction_score"].median():.3f}', linewidth=2)
ax.set_xlabel('Contradiction Score')
ax.set_ylabel('Number of Posts')
ax.set_title('Text-Image Contradiction Distribution', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 7.3: Contradiction Score vs Suspicion Score
ax = axes[0, 2]
scatter = ax.scatter(merged['contradiction_score'], merged['suspicion_score'], 
                     c=merged['is_suspicious'].astype(int), cmap='RdYlGn_r',
                     alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Contradiction Score')
ax.set_ylabel('Suspicion Score')
ax.set_title(f'Contradiction vs Suspicion\n(r={corr_contradiction_suspicion:.3f})', fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Is Suspicious')

# 7.4: Contradiction scores by dominant modality
ax = axes[1, 0]
modality_contradiction = merged.groupby('dominant_modality')['contradiction_score'].mean()
colors = {'text': '#3498db', 'image': '#e74c3c', 'meta': '#2ecc71'}
bars = ax.bar(range(len(modality_contradiction)), modality_contradiction.values,
              color=[colors.get(m, 'gray') for m in modality_contradiction.index],
              edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(modality_contradiction)))
ax.set_xticklabels(modality_contradiction.index, rotation=45)
ax.set_ylabel('Average Contradiction Score')
ax.set_title('Contradiction by Dominant Modality', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 7.5: High contradiction posts - fusion weights heatmap
ax = axes[1, 1]
high_contradiction = merged.nlargest(20, 'contradiction_score')
fusion_matrix = high_contradiction[['text_weight', 'image_weight', 'meta_weight']].values
im = ax.imshow(fusion_matrix.T, aspect='auto', cmap='YlOrRd')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Text', 'Image', 'Meta'])
ax.set_xlabel('Post Index (sorted by contradiction)')
ax.set_title('Fusion Weights Heatmap\n(Top 20 High Contradiction Posts)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Weight')

# 7.6: Detection method contributions
ax = axes[1, 2]
method_names = ['Isolation\nForest', 'DBSCAN\nOutliers', 'High\nDistance', 'Deception\nCluster']
method_contributions = [
    high_conf['iso_forest_flag'].mean(),
    high_conf['dbscan_outlier'].mean(),
    high_conf['high_distance'].mean(),
    high_conf['in_deception_cluster'].mean()
]

colors_methods = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
ax.bar(method_names, method_contributions, color=colors_methods, edgecolor='black', alpha=0.8)
ax.set_ylabel('Detection Rate')
ax.set_title('Detection Method Contributions', fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(method_contributions):
    ax.text(i, v + 0.03, f'{v*100:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'explainability_dashboard_with_fusion_and_contradiction.png', 
           dpi=150, bbox_inches='tight')
print(f"✅ Saved enhanced visualization dashboard")

# ============================================================================
# STEP 8: SAVE ALL ARTIFACTS
# ============================================================================
print("\n[STEP 8] Saving all artifacts...")

# Save fusion weights with contradiction scores for all posts
fusion_with_contradiction = fusion_df.merge(
    df_preprocessed[['post_id', 'contradiction_score']], 
    on='post_id', 
    how='left'
)
fusion_with_contradiction.to_csv(output_dir / 'fusion_and_contradiction_all_posts.csv', index=False)

# Save suspicious phrases
pd.DataFrame(top_phrases, columns=['phrase', 'score']).to_csv(
    output_dir / 'suspicious_phrases.csv', index=False
)

# Save contradiction analysis
contradiction_analysis = pd.DataFrame({
    'metric': ['Mean Suspicious', 'Mean Normal', 'Std Suspicious', 'Std Normal', 
               'Correlation with Suspicion', 'P-value'],
    'value': [
        suspicious_fusion['contradiction_score'].mean(),
        normal_fusion['contradiction_score'].mean(),
        suspicious_fusion['contradiction_score'].std(),
        normal_fusion['contradiction_score'].std(),
        corr_contradiction_suspicion,
        p_value
    ]
})
contradiction_analysis.to_csv(output_dir / 'contradiction_analysis.csv', index=False)

print(f"✅ Saved all artifacts to {output_dir}/")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("✅ ENHANCED EXPLAINABILITY WITH FUSION WEIGHTS & CONTRADICTION SCORES COMPLETE!")
print("="*80)

print(f"\n📊 Summary Statistics:")
print(f"   Total Posts Analyzed: {len(merged)}")
print(f"   High-Confidence Suspicious: {len(high_conf)}")
print(f"   Suspicious Phrases Identified: {len(top_phrases)}")

print(f"\n📊 Contradiction Score Insights:")
print(f"   Average (Suspicious): {suspicious_fusion['contradiction_score'].mean():.4f}")
print(f"   Average (Normal): {normal_fusion['contradiction_score'].mean():.4f}")
print(f"   Correlation with Suspicion: {corr_contradiction_suspicion:.4f}")
print(f"   High Contradiction Posts (>0.5): {(merged['contradiction_score'] > 0.5).sum()}")

print(f"\n📊 Fusion Weight Insights:")
print(f"   Average Text Weight (Suspicious): {suspicious_fusion['text_weight'].mean():.4f}")
print(f"   Average Image Weight (Suspicious): {suspicious_fusion['image_weight'].mean():.4f}")
print(f"   Average Meta Weight (Suspicious): {suspicious_fusion['meta_weight'].mean():.4f}")

print(f"\n📁 Enhanced Explainability Results:")
print(f"   {output_dir}/")
print(f"   ├── explainability_dashboard_with_fusion_and_contradiction.png")
print(f"   ├── post_explanations_with_fusion_and_contradiction.csv")
print(f"   ├── fusion_and_contradiction_all_posts.csv")
print(f"   ├── contradiction_analysis.csv")
print(f"   └── suspicious_phrases.csv")

print("\n🎯 Key Features:")
print("   ✅ Fusion weights show which modality (text/image/meta) drove each detection")
print("   ✅ Contradiction scores quantify text-image mismatch")
print("   ✅ High contradiction + high suspicion = strong fake news indicator")
print("   ✅ Use post_explanations_with_fusion_and_contradiction.csv for analyst review!")

# Print some example high-contradiction cases
print("\n📋 Top 5 Posts with Highest Contradiction Scores:")
top_contradiction = merged.nlargest(5, 'contradiction_score')[
    ['post_id', 'contradiction_score', 'suspicion_score', 'is_suspicious', 'dominant_modality']
]
print(top_contradiction.to_string(index=False))