"""
FULL EXPLAINABILITY LAYER FOR SUSPICIOUS CONTENT DETECTION
==========================================================
Generates human-readable explanations for suspicious posts
using HetGNN influence, text attribution, and modality fusion weights.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import networkx as nx
from datetime import datetime
import warnings
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("FULL EXPLAINABILITY LAYER FOR SUSPICIOUS CONTENT DETECTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📦 Loading post data and detection results...")

df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")\
       .drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(int)

detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
high_conf  = pd.read_csv("suspicious_detection_results/high_confidence_suspicious.csv")

# Merge suspicion_score into df if missing
if 'suspicion_score' not in df.columns:
    score_col = 'suspicion_score'
    id_col = 'post_id' if 'post_id' in detections.columns else detections.columns[0]

    if score_col in detections.columns:
        detections[id_col] = detections[id_col].astype(int)
        df = df.merge(
            detections[[id_col, score_col]].drop_duplicates(subset=id_col),
            left_on='post_id',
            right_on=id_col,
            how='left'
        )
        df['suspicion_score'] = df['suspicion_score'].fillna(0.0)
        print(f"✅ Merged suspicion_score from detections ({df['suspicion_score'].gt(0).sum()} non-zero scores)")
    else:
        raise KeyError(
            "'suspicion_score' not found in either df or detections.\n"
            f"  df columns: {df.columns.tolist()}\n"
            f"  detections columns: {detections.columns.tolist()}"
        )

print(f"✅ Loaded {len(df)} posts")
print(f"✅ Loaded {len(detections)} detection results")

# ============================================================================
# 2. LOAD HETEROGENEOUS GRAPH AND PRETRAINED HETGNN
# ============================================================================
print("\n📦 Loading HetGNN model and graph...")

node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# Normalize features
for ntype, features in node_features.items():
    features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    node_features[ntype] = F.normalize(features, p=2, dim=1).to(device)

edge_dict = {k: (ei.to(device), ew.to(device)) for k, (ei, ew) in edge_dict.items()}
node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}

# Load trained HetGNN
hetgnn_ckpt = "trained_models/best_model.pt"
checkpoint = torch.load(hetgnn_ckpt, map_location=device)

# Extract model state and hyperparameters from checkpoint
state_gnn = checkpoint.get("model_state_dict", checkpoint)

# Try to get the actual model config from checkpoint
# If the training script saved hyperparameters, use them; otherwise use defaults
if isinstance(checkpoint, dict) and 'config' in checkpoint:
    # Best case: checkpoint has saved config
    config = checkpoint['config']
    node_dims = config.get('node_dims', {"post": 128, "user": 6, "hashtag": 64, "community": 32, "deception_cluster": 32})
    hidden_dim = config.get('hidden_dim', 256)
    num_layers = config.get('num_layers', 3)
    num_classes = config.get('num_classes', 2)
    relation_types = config.get('relation_types', [
        ('user', 'creates', 'post'),
        ('post', 'contains', 'hashtag'),
        ('post', 'belongs_to', 'community'),
        ('post', 'flagged_in', 'deception_cluster'),
        ('hashtag', 'cooccurs_with', 'hashtag'),
        ('user', 'interacts_with', 'user'),
        ('deception_cluster', 'colludes_with', 'deception_cluster')
    ])
else:
    # Fallback: infer from loaded graph features (more reliable than hardcoding)
    node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}
    
    # Default hyperparameters (these should match what was used in training)
    hidden_dim = 256
    num_layers = 3
    num_classes = 2
    relation_types = [
        ('user', 'creates', 'post'),
        ('post', 'contains', 'hashtag'),
        ('post', 'belongs_to', 'community'),
        ('post', 'flagged_in', 'deception_cluster'),
        ('hashtag', 'cooccurs_with', 'hashtag'),
        ('user', 'interacts_with', 'user'),
        ('deception_cluster', 'colludes_with', 'deception_cluster')
    ]
    print(f"⚠️  Using inferred node_dims: {node_dims}")
    print(f"⚠️  If HetGNN loading fails, check that these match your training config")

# Reconstruct model with correct architecture
gnn_model = TemporalHeterogeneousGNN(
    node_dims=node_dims, 
    hidden_dim=hidden_dim, 
    num_layers=num_layers,
    relation_types=relation_types, 
    num_classes=num_classes
).to(device)

# Load trained weights
missing_keys, unexpected_keys = gnn_model.load_state_dict(state_gnn, strict=False)
gnn_model.eval()

print("✅ HetGNN loaded from trained checkpoint!")
print(f"   Node dims: {node_dims}")
print(f"   Hidden dim: {hidden_dim}, Layers: {num_layers}, Classes: {num_classes}")
if missing_keys or unexpected_keys:
    print(f"⚠️  Missing keys: {missing_keys}")
    print(f"⚠️  Unexpected keys: {unexpected_keys}")

# ============================================================================
# 3. LOAD EMOTION-AWARE MODEL (ONCE - REUSED THROUGHOUT)
# ============================================================================
from rough_work import EmotionAwareFakeNewsDetector
import torch.nn as nn

print("\n📦 Loading EmotionAwareFakeNewsDetector (with layer patching)...")

state = torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=device)

# Construct with correct top-level args
emotion_model = EmotionAwareFakeNewsDetector(
    d_text=128, d_image=1024, d_meta=128, d_common=256,
    vad_dim=3, meta_affective_dim=128, mismatch_dim=128,
    temporal_hidden=64, num_classes=1
).to(device)

# -------------------------------
# Rebuild layers to EXACT checkpoint dims
# -------------------------------
def rebuild_sequential(state_dict, prefix, indices, dropout=0.3):
    """Rebuild an nn.Sequential from checkpoint weights given layer indices of Linear layers."""
    layers = []
    for i, idx in enumerate(indices):
        w = state_dict[f"{prefix}.{idx}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(indices) - 1:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)

# 1) Patch classifier:  [256,449] -> [128,256] -> [1,128]
emotion_model.classifier = rebuild_sequential(
    state, "classifier", [0, 3, 6]
).to(device)

# 2) Patch gating_network to match exact checkpoint structure
#    Checkpoint: 0=Linear[194->256], 1=ReLU, 2=Dropout, 3=Linear[256->128], 4=ReLU, 5=Linear[128->3]
gn_layers = []
w0 = state["fusion.emotion_gate.gating_network.0.weight"]
gn_layers.append(nn.Linear(w0.shape[1], w0.shape[0]))  # index 0
gn_layers.append(nn.ReLU())                              # index 1
gn_layers.append(nn.Dropout(0.3))                        # index 2
w3 = state["fusion.emotion_gate.gating_network.3.weight"]
gn_layers.append(nn.Linear(w3.shape[1], w3.shape[0]))  # index 3
gn_layers.append(nn.ReLU())                              # index 4
w5 = state["fusion.emotion_gate.gating_network.5.weight"]
gn_layers.append(nn.Linear(w5.shape[1], w5.shape[0]))  # index 5
emotion_model.fusion_layer.emotion_gate.gating_network = nn.Sequential(*gn_layers).to(device)

# 3) Patch mismatch_encoder:  [256,131] -> [128,256] -> [128,128]
emotion_model.fusion_layer.emotion_gate.mismatch_generator.mismatch_encoder = rebuild_sequential(
    state,
    "fusion.emotion_gate.mismatch_generator.mismatch_encoder",
    [0, 3, 6]
).to(device)

# -------------------------------
# Rename checkpoint keys: fusion. → fusion_layer.
# -------------------------------
renamed_state = {}
for k, v in state.items():
    if k.startswith("fusion."):
        renamed_state[k.replace("fusion.", "fusion_layer.")] = v
    else:
        renamed_state[k] = v

# -------------------------------
# STRICT LOAD (ONCE - this is the only load of this model)
# -------------------------------
missing, unexpected = emotion_model.load_state_dict(renamed_state, strict=True)
assert not missing and not unexpected, f"Mismatch!\nMissing: {missing}\nUnexpected: {unexpected}"

emotion_model.eval()
print("✅ EmotionAwareFakeNewsDetector loaded EXACTLY as trained")

# ============================================================================
# 4. TEXT ATTRIBUTION
# ============================================================================
print("\n📦 Computing text attribution scores...")

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

text_col = 'post_text' if 'post_text' in df.columns else 'text'
word_scores, vectorizer = compute_text_attribution(df[text_col], df['suspicion_score'])
print(f"✅ Computed word importance scores")

# ============================================================================
# 5. EXTRACT FUSION WEIGHTS (using the SAME emotion_model loaded above)
# ============================================================================
print("\n📦 Extracting fusion weights from loaded emotion model...")

with open("Dataset/twitter/image_embeddings_cache.pkl", "rb") as f:
    image_embeddings = pickle.load(f)["image_embeddings"]

text_embeddings = torch.tensor(np.array(df["semantic_vector"].tolist()), dtype=torch.float32)
metadata_embeddings = torch.load("metadata_user_sequence_embeddings.pt")
if metadata_embeddings.dim() == 3:
    metadata_embeddings = metadata_embeddings.squeeze(1)
vad_data = torch.load("Dataset/twitter/prepared_vad_data.pt")

# Align embedding sizes with df (CRITICAL for explainability correctness)
N = len(df)

# --- Image embeddings ---
if len(image_embeddings) < N:
    pad = torch.zeros(N - len(image_embeddings), image_embeddings.shape[1])
    image_embeddings = torch.cat([image_embeddings, pad], dim=0)
else:
    image_embeddings = image_embeddings[:N]

# --- Metadata embeddings ---
if len(metadata_embeddings) < N:
    pad = torch.zeros(N - len(metadata_embeddings), metadata_embeddings.shape[1])
    metadata_embeddings = torch.cat([metadata_embeddings, pad], dim=0)
else:
    metadata_embeddings = metadata_embeddings[:N]

# --- Text embeddings sanity check ---
assert len(text_embeddings) == N, "Text embeddings must match df length"

# --- Create dataset and extract fusion weights ---
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
            "post_id": self.df.iloc[idx]["post_id"]
        }

dataset = FusionWeightDataset(df, text_embeddings, image_embeddings, metadata_embeddings, vad_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

all_fusion_weights = []
all_post_ids = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting fusion weights"):
        h_text = batch['text'].to(device)
        h_image = batch['image'].to(device)
        h_meta = batch['meta'].to(device)
        vad_text = batch['vad_text'].to(device)
        vad_image = batch['vad_image'].to(device)
        affective_meta = batch['affective_meta'].to(device)

        logits, intermediates = emotion_model(
            h_text, h_image, h_meta,
            vad_text=vad_text, vad_image=vad_image,
            affective_meta=affective_meta
        )
        all_fusion_weights.append(intermediates['emotion_weights'].cpu())
        all_post_ids.extend(batch['post_id'].tolist())

fusion_tensor = torch.cat(all_fusion_weights, dim=0)
fusion_df = pd.DataFrame({
    'post_id': all_post_ids,
    'text_weight': fusion_tensor[:,0].numpy(),
    'image_weight': fusion_tensor[:,1].numpy(),
    'meta_weight': fusion_tensor[:,2].numpy()
})
fusion_df['dominant_modality'] = [ ['text','image','meta'][i] for i in fusion_tensor.argmax(dim=1).numpy() ]

print(f"✅ Fusion weights extracted from the same model used for detection")

# ============================================================================
# 6. HUMAN-READABLE EXPLANATIONS
# ============================================================================
print("\n📦 Generating human-readable explanations...")

class HumanReadableExplainer:
    def __init__(self, hetgnn_explainer, word_scores, fusion_df, user_risk_scores=None):
        self.hetgnn_explainer = hetgnn_explainer
        self.word_scores = word_scores
        self.fusion_df = fusion_df.set_index('post_id')
        self.user_risk_scores = user_risk_scores or {}

    def generate_post_report(self, post_row, top_k_influencers=5, top_k_words=5):
        post_id = post_row['post_id']
        username = post_row.get('username','unknown')
        suspicion_score = post_row.get('suspicion_score',0)

        # HetGNN influence narrative
        network_narrative = self.hetgnn_explainer.generate_explanation(post_row, top_k=top_k_influencers)['narrative']

        # Textual highlights
        text = str(post_row.get('post_text', post_row.get('text','')))
        words = [w.lower().strip('.,!?') for w in text.split()]
        highlighted_words = [(w,self.word_scores[w]) for w in words if w in self.word_scores]
        highlighted_words = sorted(highlighted_words, key=lambda x:x[1], reverse=True)[:top_k_words]
        text_narrative = f"Key suspicious words or phrases include: {', '.join([w for w,_ in highlighted_words])}. " if highlighted_words else ""

        # Fusion weights
        fusion_narrative = ""
        if post_id in self.fusion_df.index:
            row_fusion = self.fusion_df.loc[post_id]
            dominant_modality = row_fusion[['text_weight','image_weight','meta_weight']].idxmax().replace('_weight','')
            weight_val = row_fusion[dominant_modality + '_weight']
            fusion_narrative = f"The detection is mostly driven by {dominant_modality} signals ({weight_val:.0%} contribution). "

        # User risk
        user_narrative = ""
        if username in self.user_risk_scores:
            risk = self.user_risk_scores[username]['overall_risk']
            if risk>0.7:
                user_narrative = "The account exhibits high-risk behavior patterns. "
            elif risk>0.4:
                user_narrative = "The account shows moderate-risk behavior patterns. "
            else:
                user_narrative = "The account appears low-risk. "

        full_narrative = (
            f"Post ID {post_id} has a suspicion score of {suspicion_score:.2f}. "
            f"{network_narrative} {text_narrative}{fusion_narrative}{user_narrative}"
        )

        return {'post_id': post_id, 'username': username, 'suspicion_score': suspicion_score, 'report': full_narrative}

    def explain_posts(self, posts_df):
        return pd.DataFrame([self.generate_post_report(row) for _,row in posts_df.iterrows()])

# Real HetGNN explainer using the trained model
class HetGNNNarrativeExplainer:
    def __init__(self, gnn_model, node_features, edge_dict, node_mappings, df):
        """
        Real GNN-based explainer that traces network influence.
        
        Args:
            gnn_model: Trained TemporalHeterogeneousGNN
            node_features: Dict of node type -> feature tensors
            edge_dict: Dict of edge type -> (edge_index, edge_weight)
            node_mappings: Dict with post_to_idx, user_to_idx, etc.
            df: DataFrame with post metadata
        """
        self.gnn_model = gnn_model
        self.node_features = node_features
        self.edge_dict = edge_dict
        self.node_mappings = node_mappings
        self.df = df.set_index('post_id')
        
        # Extract post embeddings from GNN (do this once for all posts)
        with torch.no_grad():
            self.post_embeddings = self._get_post_embeddings()
    
    def _get_post_embeddings(self):
        """Extract post node embeddings from the trained GNN."""
        # Run forward pass through the trained GNN to get learned embeddings
        try:
            # Most HetGNNs return embeddings via forward() or have an encode() method
            # Try forward first (most common)
            if hasattr(self.gnn_model, 'get_embeddings'):
                # Some models have a dedicated embedding extraction method
                embeddings = self.gnn_model.get_embeddings(self.node_features, self.edge_dict)
                return embeddings.get('post', embeddings) if isinstance(embeddings, dict) else embeddings
            else:
                # Standard approach: forward pass returns logits, but intermediate layers have embeddings
                # For explainability, we want the learned post representations, not raw features
                # The GNN transforms raw features through multiple layers
                with torch.no_grad():
                    # If the model has a forward that returns embeddings
                    output = self.gnn_model(self.node_features, self.edge_dict)
                    # output could be logits, embeddings dict, or tuple (embeddings, logits)
                    if isinstance(output, dict) and 'post' in output:
                        return output['post']
                    elif isinstance(output, tuple):
                        # Usually (embeddings_dict, logits)
                        embeddings, _ = output
                        if isinstance(embeddings, dict):
                            return embeddings.get('post', self.node_features['post'])
                        return embeddings
                    else:
                        # Fallback: if we can't extract embeddings, use the transformed features
                        # This still uses the GNN's learned transformations from the input layer
                        return self.node_features['post']
        except Exception as e:
            print(f"⚠️  Could not extract GNN embeddings: {e}")
            print(f"   Using input node features as fallback")
            # Even the fallback uses features that went through the graph construction,
            # just not the learned GNN transformations
            return self.node_features['post']
    
    def generate_explanation(self, post_row, top_k=5):
        """
        Generate a network-based explanation for a suspicious post.
        
        Args:
            post_row: DataFrame row for the post
            top_k: Number of top influencers to identify
        
        Returns:
            Dict with 'narrative' key containing the explanation
        """
        post_id = post_row['post_id']
        
        # Get post index in the graph
        # node_mappings could be {'post': {id: idx}, 'user': {id: idx}} 
        # or {'post_to_idx': {id: idx}, 'user_to_idx': {id: idx}}
        post_mapping = self.node_mappings.get('post_to_idx') or self.node_mappings.get('post')
        
        if post_mapping is None or post_id not in post_mapping:
            return {'narrative': "This post is not in the network graph."}
        
        post_idx = post_mapping[post_id]
        
        # --- 1. Find connected users ---
        connected_users = self._get_connected_users(post_idx)
        
        # --- 2. Find connected hashtags ---
        connected_hashtags = self._get_connected_hashtags(post_idx)
        
        # --- 3. Find deception cluster assignment ---
        deception_cluster = self._get_deception_cluster(post_idx)
        
        # --- 4. Find community assignment ---
        community = self._get_community(post_idx)
        
        # --- 5. Build narrative ---
        narrative_parts = []
        
        # User connections
        if connected_users:
            user_info = connected_users[:min(2, len(connected_users))]  # Top 2 users
            user_names = [self._get_username(uid) for uid in user_info]
            if len(user_names) == 1:
                narrative_parts.append(f"Posted by user {user_names[0]}")
            else:
                narrative_parts.append(f"Connected to users {', '.join(user_names)}")
        
        # Deception cluster
        if deception_cluster is not None:
            narrative_parts.append(f"flagged in deception cluster {deception_cluster}")
        
        # Hashtags
        if connected_hashtags:
            hashtag_info = connected_hashtags[:min(3, len(connected_hashtags))]
            hashtag_names = [self._get_hashtag_text(hid) for hid in hashtag_info]
            if hashtag_names:
                narrative_parts.append(f"uses hashtags: {', '.join(hashtag_names)}")
        
        # Community
        if community is not None:
            narrative_parts.append(f"belongs to community {community}")
        
        # Combine narrative
        if narrative_parts:
            narrative = "This post is " + ", ".join(narrative_parts) + ". "
        else:
            narrative = "This post has limited network connections. "
        
        # Add network influence score if available
        if self.post_embeddings is not None and post_idx < len(self.post_embeddings):
            embedding_norm = self.post_embeddings[post_idx].norm().item()
            if embedding_norm > 5.0:  # Threshold for "high influence"
                narrative += "It shows high network influence. "
        
        return {'narrative': narrative}
    
    def _get_connected_users(self, post_idx):
        """Find users connected to this post via 'creates' edge."""
        edge_type = ('user', 'creates', 'post')
        if edge_type not in self.edge_dict:
            return []
        
        edge_index, _ = self.edge_dict[edge_type]
        # edge_index[0] = user indices, edge_index[1] = post indices
        # Find all users where edge_index[1] == post_idx
        mask = edge_index[1] == post_idx
        user_indices = edge_index[0][mask].cpu().tolist()
        return user_indices
    
    def _get_connected_hashtags(self, post_idx):
        """Find hashtags connected to this post via 'contains' edge."""
        edge_type = ('post', 'contains', 'hashtag')
        if edge_type not in self.edge_dict:
            return []
        
        edge_index, _ = self.edge_dict[edge_type]
        # edge_index[0] = post indices, edge_index[1] = hashtag indices
        mask = edge_index[0] == post_idx
        hashtag_indices = edge_index[1][mask].cpu().tolist()
        return hashtag_indices
    
    def _get_deception_cluster(self, post_idx):
        """Find deception cluster this post is flagged in."""
        edge_type = ('post', 'flagged_in', 'deception_cluster')
        if edge_type not in self.edge_dict:
            return None
        
        edge_index, _ = self.edge_dict[edge_type]
        mask = edge_index[0] == post_idx
        if mask.any():
            cluster_idx = edge_index[1][mask][0].item()
            return cluster_idx
        return None
    
    def _get_community(self, post_idx):
        """Find community this post belongs to."""
        edge_type = ('post', 'belongs_to', 'community')
        if edge_type not in self.edge_dict:
            return None
        
        edge_index, _ = self.edge_dict[edge_type]
        mask = edge_index[0] == post_idx
        if mask.any():
            community_idx = edge_index[1][mask][0].item()
            return community_idx
        return None
    
    def _get_username(self, user_idx):
        """Get username from user index."""
        # Reverse lookup: find username from user index
        user_mapping = self.node_mappings.get('user_to_idx') or self.node_mappings.get('user')
        if user_mapping is None:
            return f"user_{user_idx}"
        
        idx_to_user = {v: k for k, v in user_mapping.items()}
        if user_idx in idx_to_user:
            return idx_to_user[user_idx]
        return f"user_{user_idx}"
    
    def _get_hashtag_text(self, hashtag_idx):
        """Get hashtag text from hashtag index."""
        hashtag_mapping = self.node_mappings.get('hashtag_to_idx') or self.node_mappings.get('hashtag')
        if hashtag_mapping is None:
            return f"#{hashtag_idx}"
        
        idx_to_hashtag = {v: k for k, v in hashtag_mapping.items()}
        if hashtag_idx in idx_to_hashtag:
            return idx_to_hashtag[hashtag_idx]
        return f"#{hashtag_idx}"

hetgnn_explainer = HetGNNNarrativeExplainer(
    gnn_model=gnn_model,
    node_features=node_features,
    edge_dict=edge_dict,
    node_mappings=node_mappings,
    df=df
)
explainer = HumanReadableExplainer(hetgnn_explainer, word_scores, fusion_df)

top_posts = high_conf.nlargest(20,'suspicion_score')
reports_df = explainer.explain_posts(top_posts)

# ==========================
# 8️⃣ Add final Fake/Real verdict
# ==========================
def classify_post(suspicion_score, fusion_weights=None, network_embedding_norm=None):
    """
    Return 'Fake' or 'Real' based on suspicion score and optional cues.
    """
    # Primary threshold
    if suspicion_score >= 0.6:
        verdict = "Fake"
    elif suspicion_score < 0.3:
        verdict = "Real"
    else:
        # Tie-breakers using dominant modality
        if fusion_weights is not None:
            if fusion_weights['text_weight'] > 0.5:
                verdict = "Fake"
            else:
                verdict = "Real"
        else:
            verdict = "Real"

    # Optional: network influence override
    if network_embedding_norm is not None and network_embedding_norm > 5.0:
        verdict = "Fake"

    return verdict



# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
output_dir = Path("explainability_results")
output_dir.mkdir(exist_ok=True)

reports_df.to_csv(output_dir / "human_readable_reports.csv", index=False)
fusion_df.to_csv(output_dir / "fusion_weights_all_posts.csv", index=False)

# Compute verdicts for all posts in reports_df
verdicts = []
fusion_df_indexed = fusion_df.set_index('post_id')
for _, row in reports_df.iterrows():
    post_id = row['post_id']
    
    # Network norm
    network_norm = None
    if hetgnn_explainer.post_embeddings is not None:
        post_mapping = hetgnn_explainer.node_mappings.get('post_to_idx') or hetgnn_explainer.node_mappings.get('post')
        post_idx = post_mapping.get(post_id, None)
        if post_idx is not None:
            network_norm = hetgnn_explainer.post_embeddings[post_idx].norm().item()
    
    # Fusion weights
    if post_id in fusion_df_indexed.index:
        fusion_weights = fusion_df_indexed.loc[post_id]
    else:
        fusion_weights = pd.Series({'text_weight':0,'image_weight':0,'meta_weight':0})
    
    verdicts.append(classify_post(
        row['suspicion_score'],
        fusion_weights=fusion_weights,
        network_embedding_norm=network_norm
    ))

reports_df['verdict'] = verdicts

# Save final explainability report
reports_df.to_csv(output_dir / "human_readable_reports.csv", index=False)
fusion_df.to_csv(output_dir / "fusion_weights_all_posts.csv", index=False)

print(f"✅ Human-readable reports with final verdict saved to {output_dir}/human_readable_reports.csv")
print(f"✅ Fusion weights saved to {output_dir}/fusion_weights_all_posts.csv")
