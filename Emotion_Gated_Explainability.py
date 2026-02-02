"""
COMPREHENSIVE EXPLAINABILITY LAYER
===================================
Combines:
- Fusion weights (which modality dominated)
- Contradiction scores (text-image mismatch)
- Emotion-gated mechanism insights (congruence, mismatch, temporal, mixed affect)
- Psychologically-grounded explanations

This provides the most complete explainability for fake news detection.
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
print("COMPREHENSIVE EXPLAINABILITY WITH EMOTION-GATED INSIGHTS")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL DATA SOURCES
# ============================================================================
print("\n[STEP 1] Loading all data sources...")

# Load preprocessed data with contradiction scores
df_preprocessed = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
print(f"✅ Loaded preprocessed data: {len(df_preprocessed)} posts")

# Load full dataframe
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(int)

# Load detection results
detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
high_conf = pd.read_csv("suspicious_detection_results/high_confidence_suspicious.csv")

print(f"✅ Loaded {len(detections)} detections, {len(high_conf)} high-confidence")

# ============================================================================
# STEP 2: EXTRACT EMOTION-GATED MECHANISM INSIGHTS
# ============================================================================
print("\n[STEP 2] Extracting emotion-gated mechanism insights...")

# Load embeddings
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

# Load trained emotion-aware model
print("Loading emotion-aware model...")
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
class ComprehensiveDataset(Dataset):
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

dataset = ComprehensiveDataset(df, text_embeddings, image_embeddings, metadata_embeddings, vad_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract ALL emotion-gated insights
print("Extracting comprehensive emotion insights...")
all_emotion_weights = []
all_congruence = []
all_mismatch_magnitude = []
all_mixed_affect = []
all_vad_text = []
all_vad_image = []
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
        
        # Extract all emotion signals
        emotion_weights = intermediates['emotion_weights']  # [batch, 3]
        congruence = intermediates['congruence']  # [batch, 1]
        v_mismatch = intermediates['v_mismatch']  # [batch, mismatch_dim]
        mixed_affect = intermediates['mixed_affect_score']  # [batch, 1]
        
        # Compute mismatch magnitude
        mismatch_magnitude = torch.norm(v_mismatch, dim=1, keepdim=True)
        
        all_emotion_weights.append(emotion_weights.cpu())
        all_congruence.append(congruence.cpu())
        all_mismatch_magnitude.append(mismatch_magnitude.cpu())
        all_mixed_affect.append(mixed_affect.cpu())
        all_vad_text.append(vad_text.cpu())
        all_vad_image.append(vad_image.cpu())
        all_post_ids.extend(batch['post_id'].tolist())

# Concatenate all
emotion_weights_tensor = torch.cat(all_emotion_weights, dim=0)
congruence_tensor = torch.cat(all_congruence, dim=0).squeeze()
mismatch_magnitude_tensor = torch.cat(all_mismatch_magnitude, dim=0).squeeze()
mixed_affect_tensor = torch.cat(all_mixed_affect, dim=0).squeeze()
vad_text_tensor = torch.cat(all_vad_text, dim=0)
vad_image_tensor = torch.cat(all_vad_image, dim=0)

# Create comprehensive emotion insights dataframe
emotion_insights_df = pd.DataFrame({
    'post_id': all_post_ids,
    'text_weight': emotion_weights_tensor[:, 0].numpy(),
    'image_weight': emotion_weights_tensor[:, 1].numpy(),
    'meta_weight': emotion_weights_tensor[:, 2].numpy(),
    'emotional_congruence': congruence_tensor.numpy(),
    'mismatch_magnitude': mismatch_magnitude_tensor.numpy(),
    'mixed_affect_score': mixed_affect_tensor.numpy(),
    'vad_text_valence': vad_text_tensor[:, 0].numpy(),
    'vad_text_arousal': vad_text_tensor[:, 1].numpy(),
    'vad_text_dominance': vad_text_tensor[:, 2].numpy(),
    'vad_image_valence': vad_image_tensor[:, 0].numpy(),
    'vad_image_arousal': vad_image_tensor[:, 1].numpy(),
    'vad_image_dominance': vad_image_tensor[:, 2].numpy(),
})

# Add dominant modality
modality_names = ['text', 'image', 'meta']
emotion_insights_df['dominant_modality'] = [
    modality_names[idx] for idx in emotion_weights_tensor.argmax(dim=1).numpy()
]

print(f"✅ Extracted emotion insights for {len(emotion_insights_df)} posts")

# ============================================================================
# STEP 3: MERGE ALL DATA TOGETHER
# ============================================================================
print("\n[STEP 3] Merging all data sources...")

# Merge with detections
detections = detections.merge(emotion_insights_df, on='post_id', how='left')
high_conf = high_conf.merge(emotion_insights_df, on='post_id', how='left')

# Merge with full df
merged = detections.merge(df, on='post_id', how='left')

# Merge contradiction scores
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
    merged['contradiction_score'] = merged['contradiction_score_y'].fillna(merged['contradiction_score_x'])
    merged = merged.drop(columns=['contradiction_score_x', 'contradiction_score_y'])
    
    high_conf['contradiction_score'] = high_conf['contradiction_score_y'].fillna(high_conf['contradiction_score_x'])
    high_conf = high_conf.drop(columns=['contradiction_score_x', 'contradiction_score_y'])
    print(f"✅ Cleaned up duplicate columns")

print(f"✅ Comprehensive merge complete")

# ============================================================================
# STEP 4: COMPUTE PSYCHOLOGICAL INSIGHTS
# ============================================================================
print("\n[STEP 4] Computing psychological insights...")

# Emotional inconsistency index
merged['emotional_inconsistency'] = (
    0.5 * (1 - merged['emotional_congruence']) +
    0.3 * merged['mismatch_magnitude'] / merged['mismatch_magnitude'].max() +
    0.2 * merged['mixed_affect_score']
)

# Deception risk score (composite)
merged['deception_risk_score'] = (
    0.3 * merged['suspicion_score'] +
    0.25 * merged.get('contradiction_score', 0) +
    0.25 * merged['emotional_inconsistency'] +
    0.2 * (1 - merged['emotional_congruence'])
)

# Emotional manipulation indicator
merged['emotional_manipulation'] = (
    (merged['mixed_affect_score'] > 0.7) & 
    (merged['emotional_congruence'] < -0.2)
).astype(float)

# Analyze by detection category
print("\n📊 Emotional Insights by Detection Category:")
for category in ['Suspicious', 'Normal']:
    is_suspicious = category == 'Suspicious'
    subset = merged[merged['is_suspicious'] == is_suspicious]
    
    print(f"\n{category} Posts:")
    print(f"   Avg Emotional Congruence: {subset['emotional_congruence'].mean():.4f}")
    print(f"   Avg Mismatch Magnitude: {subset['mismatch_magnitude'].mean():.4f}")
    print(f"   Avg Mixed Affect: {subset['mixed_affect_score'].mean():.4f}")
    print(f"   Avg Emotional Inconsistency: {subset['emotional_inconsistency'].mean():.4f}")

# ============================================================================
# STEP 5: TEXT ATTRIBUTION
# ============================================================================
print("\n[STEP 5] Computing text attribution...")

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

# ============================================================================
# STEP 6: GENERATE COMPREHENSIVE EXPLANATIONS
# ============================================================================
print("\n[STEP 6] Generating comprehensive explanations...")

def generate_comprehensive_explanation(post_row, word_scores):
    """
    Generate human-readable explanation with ALL insights:
    - Detection methods
    - Fusion weights
    - Contradiction scores
    - Emotional congruence
    - Mismatch magnitude
    - Mixed affect
    - VAD analysis
    """
    explanation = {
        'post_id': post_row['post_id'],
        'suspicion_score': post_row['suspicion_score'],
        'deception_risk_score': post_row.get('deception_risk_score', 0),
        
        # Detection methods
        'detection_methods': [],
        
        # Modality analysis
        'fusion_weights': {
            'text': float(post_row.get('text_weight', 0)),
            'image': float(post_row.get('image_weight', 0)),
            'meta': float(post_row.get('meta_weight', 0))
        },
        'dominant_modality': post_row.get('dominant_modality', 'unknown'),
        
        # Emotional analysis
        'emotional_congruence': float(post_row.get('emotional_congruence', 0)),
        'mismatch_magnitude': float(post_row.get('mismatch_magnitude', 0)),
        'mixed_affect_score': float(post_row.get('mixed_affect_score', 0)),
        'emotional_inconsistency': float(post_row.get('emotional_inconsistency', 0)),
        'emotional_manipulation': bool(post_row.get('emotional_manipulation', False)),
        
        # VAD dimensions
        'text_vad': {
            'valence': float(post_row.get('vad_text_valence', 0)),
            'arousal': float(post_row.get('vad_text_arousal', 0)),
            'dominance': float(post_row.get('vad_text_dominance', 0))
        },
        'image_vad': {
            'valence': float(post_row.get('vad_image_valence', 0)),
            'arousal': float(post_row.get('vad_image_arousal', 0)),
            'dominance': float(post_row.get('vad_image_dominance', 0))
        },
        
        # Cross-modal contradiction
        'contradiction_score': float(post_row.get('contradiction_score', 0)),
        
        # Text highlights
        'text_highlights': [],
        
        # Comprehensive summary
        'explanation_summary': '',
        'psychological_analysis': '',
        'modality_analysis': '',
        'emotion_analysis': ''
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
    
    # Text highlights
    text = str(post_row.get('post_text', post_row.get('text', '')))
    words = text.lower().split()
    highlighted_words = []
    for word in words:
        if word in word_scores and word_scores[word] > 0.3:
            highlighted_words.append((word, word_scores[word]))
    explanation['text_highlights'] = sorted(highlighted_words, key=lambda x: x[1], reverse=True)[:5]
    
    # Build comprehensive summary
    num_methods = len(explanation['detection_methods'])
    suspicion = explanation['suspicion_score']
    
    # Determine confidence level
    if num_methods >= 3 and suspicion > 0.75:
        confidence = "Very High"
    elif num_methods >= 2 and suspicion > 0.5:
        confidence = "High"
    elif num_methods >= 1 and suspicion > 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # === MAIN SUMMARY ===
    explanation['explanation_summary'] = (
        f"This post has a {confidence.lower()} confidence suspicion score of {suspicion:.2f}. "
        f"Overall deception risk: {explanation['deception_risk_score']:.2f}. "
        f"Flagged by {num_methods} detection method(s). "
    )
    
    # === EMOTIONAL ANALYSIS ===
    congruence = explanation['emotional_congruence']
    mismatch = explanation['mismatch_magnitude']
    mixed = explanation['mixed_affect_score']
    
    emotion_parts = []
    
    if congruence < -0.3:
        emotion_parts.append(
            f"⚠️ STRONG EMOTIONAL CONTRADICTION detected (congruence: {congruence:.2f}). "
            f"Text and image emotions are opposing each other."
        )
    elif congruence < 0:
        emotion_parts.append(
            f"Moderate emotional mismatch (congruence: {congruence:.2f})."
        )
    elif congruence > 0.5:
        emotion_parts.append(
            f"Text and image emotions are aligned (congruence: {congruence:.2f})."
        )
    
    if mismatch > 0.5:
        emotion_parts.append(
            f"High emotional mismatch magnitude ({mismatch:.2f}) indicates deceptive patterns."
        )
    
    if mixed > 0.7:
        emotion_parts.append(
            f"⚠️ MIXED AFFECT detected ({mixed:.2f}). "
            f"Content exhibits simultaneous opposing emotions - common in clickbait and manipulation."
        )
    
    if explanation['emotional_manipulation']:
        emotion_parts.append(
            f"🚨 Content shows signs of EMOTIONAL MANIPULATION (mixed affect + low congruence)."
        )
    
    explanation['emotion_analysis'] = ' '.join(emotion_parts)
    
    # === MODALITY ANALYSIS ===
    dominant_mod = explanation['dominant_modality']
    dominant_weight = explanation['fusion_weights'].get(dominant_mod, 0)
    
    modality_parts = []
    modality_parts.append(
        f"The '{dominant_mod}' modality contributed most ({dominant_weight:.2%}) to this detection."
    )
    
    if explanation['text_highlights'] and dominant_mod == 'text':
        top_word = explanation['text_highlights'][0][0]
        modality_parts.append(f"Key suspicious phrase: '{top_word}'.")
    elif dominant_mod == 'image':
        if explanation['contradiction_score'] > 0.5:
            modality_parts.append(
                "Visual content contradicts text and shows suspicious patterns."
            )
        else:
            modality_parts.append("Visual content shows suspicious patterns.")
    elif dominant_mod == 'meta':
        modality_parts.append("User behavior metadata indicates anomalous patterns.")
    
    explanation['modality_analysis'] = ' '.join(modality_parts)
    
    # === PSYCHOLOGICAL ANALYSIS ===
    psych_parts = []
    
    text_vad = explanation['text_vad']
    image_vad = explanation['image_vad']
    
    # Analyze text emotion
    if text_vad['valence'] < -0.5:
        text_emotion = "negative"
    elif text_vad['valence'] > 0.5:
        text_emotion = "positive"
    else:
        text_emotion = "neutral"
    
    if text_vad['arousal'] > 0.5:
        text_arousal = "high-arousal"
    else:
        text_arousal = "low-arousal"
    
    # Analyze image emotion
    if image_vad['valence'] < -0.5:
        image_emotion = "negative"
    elif image_vad['valence'] > 0.5:
        image_emotion = "positive"
    else:
        image_emotion = "neutral"
    
    if image_vad['arousal'] > 0.5:
        image_arousal = "high-arousal"
    else:
        image_arousal = "low-arousal"
    
    psych_parts.append(
        f"Text emotion: {text_emotion}, {text_arousal}. "
        f"Image emotion: {image_emotion}, {image_arousal}."
    )
    
    # Detect specific patterns
    if (text_emotion != image_emotion) and abs(text_vad['valence'] - image_vad['valence']) > 0.7:
        psych_parts.append(
            f"⚠️ VALENCE CONTRADICTION: Text is {text_emotion} but image is {image_emotion}. "
            f"This is a strong deception signal in multimodal content."
        )
    
    if abs(text_vad['arousal'] - image_vad['arousal']) > 0.7:
        psych_parts.append(
            f"Arousal mismatch: Text shows {text_arousal} while image shows {image_arousal}."
        )
    
    explanation['psychological_analysis'] = ' '.join(psych_parts)
    
    # === COMBINE ALL ===
    explanation['explanation_summary'] += (
        explanation['emotion_analysis'] + ' ' +
        explanation['modality_analysis'] + ' ' +
        explanation['psychological_analysis']
    )
    
    return explanation

# Generate comprehensive explanations
top_suspicious = high_conf.nlargest(20, 'suspicion_score')
explanations = []

for idx, row in top_suspicious.iterrows():
    exp = generate_comprehensive_explanation(row, word_scores)
    explanations.append(exp)

# Save comprehensive explanations
explanations_df = pd.DataFrame([
    {
        'post_id': exp['post_id'],
        'suspicion_score': exp['suspicion_score'],
        'deception_risk_score': exp['deception_risk_score'],
        'num_detection_methods': len(exp['detection_methods']),
        'detection_methods': '; '.join(exp['detection_methods']),
        
        # Modality
        'dominant_modality': exp['dominant_modality'],
        'text_weight': exp['fusion_weights']['text'],
        'image_weight': exp['fusion_weights']['image'],
        'meta_weight': exp['fusion_weights']['meta'],
        
        # Emotion
        'emotional_congruence': exp['emotional_congruence'],
        'mismatch_magnitude': exp['mismatch_magnitude'],
        'mixed_affect_score': exp['mixed_affect_score'],
        'emotional_inconsistency': exp['emotional_inconsistency'],
        'emotional_manipulation': exp['emotional_manipulation'],
        
        # VAD
        'text_valence': exp['text_vad']['valence'],
        'text_arousal': exp['text_vad']['arousal'],
        'text_dominance': exp['text_vad']['dominance'],
        'image_valence': exp['image_vad']['valence'],
        'image_arousal': exp['image_vad']['arousal'],
        'image_dominance': exp['image_vad']['dominance'],
        
        # Contradiction
        'contradiction_score': exp['contradiction_score'],
        
        # Text
        'top_suspicious_words': ', '.join([w for w, s in exp['text_highlights']]),
        
        # Explanations
        'comprehensive_summary': exp['explanation_summary'],
        'emotion_analysis': exp['emotion_analysis'],
        'modality_analysis': exp['modality_analysis'],
        'psychological_analysis': exp['psychological_analysis']
    }
    for exp in explanations
])

output_dir = Path("explainability_results")
output_dir.mkdir(exist_ok=True)

explanations_df.to_csv(output_dir / 'comprehensive_explanations.csv', index=False)
print(f"✅ Saved comprehensive explanations for {len(explanations)} posts")

# ============================================================================
# STEP 7: COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("\n[STEP 7] Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 7.1: Emotional Congruence Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(merged['emotional_congruence'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
ax1.set_xlabel('Emotional Congruence')
ax1.set_ylabel('Count')
ax1.set_title('Emotional Congruence Distribution', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 7.2: Mismatch Magnitude Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(merged['mismatch_magnitude'], bins=30, color='crimson', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Mismatch Magnitude')
ax2.set_ylabel('Count')
ax2.set_title('Emotional Mismatch Magnitude', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 7.3: Mixed Affect Score Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(merged['mixed_affect_score'], bins=30, color='orange', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Mixed Affect Score')
ax3.set_ylabel('Count')
ax3.set_title('Mixed Affect Distribution', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 7.4: Congruence vs Suspicion
ax4 = fig.add_subplot(gs[1, 0])
scatter = ax4.scatter(merged['emotional_congruence'], merged['suspicion_score'],
                      c=merged['is_suspicious'].astype(int), cmap='RdYlGn_r',
                      alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
ax4.set_xlabel('Emotional Congruence')
ax4.set_ylabel('Suspicion Score')
ax4.set_title('Congruence vs Suspicion', fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Is Suspicious')

# 7.5: Mismatch vs Suspicion
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(merged['mismatch_magnitude'], merged['suspicion_score'],
                      c=merged['is_suspicious'].astype(int), cmap='RdYlGn_r',
                      alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
ax5.set_xlabel('Mismatch Magnitude')
ax5.set_ylabel('Suspicion Score')
ax5.set_title('Mismatch vs Suspicion', fontweight='bold')
ax5.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax5, label='Is Suspicious')

# 7.6: Mixed Affect vs Suspicion
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(merged['mixed_affect_score'], merged['suspicion_score'],
                      c=merged['is_suspicious'].astype(int), cmap='RdYlGn_r',
                      alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
ax6.set_xlabel('Mixed Affect Score')
ax6.set_ylabel('Suspicion Score')
ax6.set_title('Mixed Affect vs Suspicion', fontweight='bold')
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Is Suspicious')

# 7.7: VAD Space (Text vs Image)
ax7 = fig.add_subplot(gs[2, 0])
suspicious_mask = merged['is_suspicious'] == True
ax7.scatter(merged.loc[~suspicious_mask, 'vad_text_valence'], 
           merged.loc[~suspicious_mask, 'vad_text_arousal'],
           c='green', alpha=0.3, s=20, label='Normal (Text)')
ax7.scatter(merged.loc[suspicious_mask, 'vad_text_valence'], 
           merged.loc[suspicious_mask, 'vad_text_arousal'],
           c='red', alpha=0.3, s=20, label='Suspicious (Text)')
ax7.set_xlabel('Valence')
ax7.set_ylabel('Arousal')
ax7.set_title('VAD Space - Text Emotions', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 7.8: Emotion Inconsistency Distribution
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(merged['emotional_inconsistency'], bins=30, color='purple', edgecolor='black', alpha=0.7)
ax8.set_xlabel('Emotional Inconsistency Index')
ax8.set_ylabel('Count')
ax8.set_title('Emotional Inconsistency Index', fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# 7.9: Deception Risk Score
ax9 = fig.add_subplot(gs[2, 2])
ax9.hist(merged['deception_risk_score'], bins=30, color='darkred', edgecolor='black', alpha=0.7)
ax9.set_xlabel('Deception Risk Score')
ax9.set_ylabel('Count')
ax9.set_title('Composite Deception Risk', fontweight='bold')
ax9.grid(axis='y', alpha=0.3)

# 7.10: Fusion Weights by Emotion Category
ax10 = fig.add_subplot(gs[3, 0])
congruent = merged[merged['emotional_congruence'] > 0.3]
contradictory = merged[merged['emotional_congruence'] < -0.3]

categories = ['Congruent', 'Contradictory']
text_weights = [congruent['text_weight'].mean(), contradictory['text_weight'].mean()]
image_weights = [congruent['image_weight'].mean(), contradictory['image_weight'].mean()]
meta_weights = [congruent['meta_weight'].mean(), contradictory['meta_weight'].mean()]

x = np.arange(len(categories))
width = 0.25

ax10.bar(x - width, text_weights, width, label='Text', color='#3498db')
ax10.bar(x, image_weights, width, label='Image', color='#e74c3c')
ax10.bar(x + width, meta_weights, width, label='Meta', color='#2ecc71')

ax10.set_xlabel('Emotion Category')
ax10.set_ylabel('Average Weight')
ax10.set_title('Fusion Weights by Emotion Category', fontweight='bold')
ax10.set_xticks(x)
ax10.set_xticklabels(categories)
ax10.legend()
ax10.grid(axis='y', alpha=0.3)

# 7.11: Correlation Matrix
ax11 = fig.add_subplot(gs[3, 1])
emotion_vars = ['emotional_congruence', 'mismatch_magnitude', 'mixed_affect_score', 
                'contradiction_score', 'suspicion_score']
corr_matrix = merged[emotion_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            ax=ax11, square=True, linewidths=1)
ax11.set_title('Emotion Metrics Correlation', fontweight='bold')

# 7.12: Emotional Manipulation Detection
ax12 = fig.add_subplot(gs[3, 2])
manipulation_counts = merged.groupby(['is_suspicious', 'emotional_manipulation']).size().unstack(fill_value=0)
manipulation_counts.plot(kind='bar', ax=ax12, color=['#3498db', '#e74c3c'], edgecolor='black')
ax12.set_xlabel('Is Suspicious')
ax12.set_ylabel('Count')
ax12.set_title('Emotional Manipulation Detection', fontweight='bold')
ax12.legend(['No Manipulation', 'Manipulation Detected'], loc='upper right')
ax12.grid(axis='y', alpha=0.3)
plt.setp(ax12.xaxis.get_majorticklabels(), rotation=0)

plt.savefig(output_dir / 'comprehensive_explainability_dashboard.png', 
           dpi=150, bbox_inches='tight')
print(f"✅ Saved comprehensive visualization dashboard")

# ============================================================================
# STEP 8: SAVE ALL ARTIFACTS
# ============================================================================
print("\n[STEP 8] Saving all artifacts...")

# Save comprehensive emotion insights for all posts
emotion_insights_with_scores = emotion_insights_df.merge(
    df_preprocessed[['post_id', 'contradiction_score']], 
    on='post_id', 
    how='left'
)
emotion_insights_with_scores.to_csv(output_dir / 'emotion_insights_all_posts.csv', index=False)

# Save emotional analysis summary
emotion_analysis_summary = pd.DataFrame({
    'metric': [
        'Avg Congruence (Suspicious)', 'Avg Congruence (Normal)',
        'Avg Mismatch (Suspicious)', 'Avg Mismatch (Normal)',
        'Avg Mixed Affect (Suspicious)', 'Avg Mixed Affect (Normal)',
        'Emotional Manipulation Rate (Suspicious)', 'Emotional Manipulation Rate (Normal)'
    ],
    'value': [
        merged[merged['is_suspicious'] == True]['emotional_congruence'].mean(),
        merged[merged['is_suspicious'] == False]['emotional_congruence'].mean(),
        merged[merged['is_suspicious'] == True]['mismatch_magnitude'].mean(),
        merged[merged['is_suspicious'] == False]['mismatch_magnitude'].mean(),
        merged[merged['is_suspicious'] == True]['mixed_affect_score'].mean(),
        merged[merged['is_suspicious'] == False]['mixed_affect_score'].mean(),
        merged[merged['is_suspicious'] == True]['emotional_manipulation'].mean(),
        merged[merged['is_suspicious'] == False]['emotional_manipulation'].mean()
    ]
})
emotion_analysis_summary.to_csv(output_dir / 'emotion_analysis_summary.csv', index=False)

# Save suspicious phrases
pd.DataFrame(top_phrases, columns=['phrase', 'score']).to_csv(
    output_dir / 'suspicious_phrases.csv', index=False
)

print(f"✅ Saved all artifacts to {output_dir}/")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("✅ COMPREHENSIVE EXPLAINABILITY LAYER COMPLETE!")
print("="*80)

print(f"\n📊 Summary Statistics:")
print(f"   Total Posts Analyzed: {len(merged)}")
print(f"   High-Confidence Suspicious: {len(high_conf)}")
print(f"   Emotional Manipulation Detected: {merged['emotional_manipulation'].sum()}")

print(f"\n📊 Emotion-Gated Insights:")
print(f"   Avg Congruence (Suspicious): {merged[merged['is_suspicious'] == True]['emotional_congruence'].mean():.4f}")
print(f"   Avg Congruence (Normal): {merged[merged['is_suspicious'] == False]['emotional_congruence'].mean():.4f}")
print(f"   Avg Mismatch (Suspicious): {merged[merged['is_suspicious'] == True]['mismatch_magnitude'].mean():.4f}")
print(f"   Avg Mismatch (Normal): {merged[merged['is_suspicious'] == False]['mismatch_magnitude'].mean():.4f}")

print(f"\n📁 Comprehensive Explainability Results:")
print(f"   {output_dir}/")
print(f"   ├── comprehensive_explainability_dashboard.png  - Complete visualizations")
print(f"   ├── comprehensive_explanations.csv             - Full explanations with ALL insights")
print(f"   ├── emotion_insights_all_posts.csv             - Emotion metrics for every post")
print(f"   ├── emotion_analysis_summary.csv               - Statistical summary")
print(f"   └── suspicious_phrases.csv                     - Text attribution scores")

print("\n🎯 Key Features:")
print("   ✅ Fusion weights (which modality dominated)")
print("   ✅ Contradiction scores (text-image mismatch)")
print("   ✅ Emotional congruence (VAD-based alignment)")
print("   ✅ Mismatch magnitude (deception signal)")
print("   ✅ Mixed affect detection (emotional manipulation)")
print("   ✅ VAD dimensions (valence, arousal, dominance)")
print("   ✅ Psychological analysis (emotion patterns)")
print("   ✅ Comprehensive deception risk scoring")

print("\n🧠 Thesis-Ready Explainability:")
print("   This provides psychologically-grounded, human-interpretable")
print("   explanations that go beyond black-box predictions.")
print("   Use comprehensive_explanations.csv for analyst review!")

# Print example explanations
print("\n📋 Example Comprehensive Explanation:")
print("-" * 80)
example = explanations[0]
print(f"Post ID: {example['post_id']}")
print(f"Suspicion: {example['suspicion_score']:.2f}")
print(f"Deception Risk: {example['deception_risk_score']:.2f}")
print(f"\n{example['explanation_summary']}")