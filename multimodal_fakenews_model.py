import re
import pandas as pd
import emoji
import spacy
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import numpy as np
import os
import gc
from tqdm import tqdm
import torchvision.models as models
from ultralytics import YOLO
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from torchvision import models, transforms as T
from PIL import Image
import h5py
from torchvision.models import vit_b_16
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pickle
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cluster import HDBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import umap
from transformers import BertTokenizerFast


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_sm")


df = pd.read_csv("Dataset/twitter/df_train_translated.csv")

def extract_hashtags(text):
    return re.findall(r'#\w+', str(text))

def extract_mentions(text):
    return re.findall(r'@\w+', str(text))

def extract_urls(text):
    return re.findall(r'http\S+|www.\S+', str(text))

def extract_emojis(text):
    return [c for c in str(text) if emoji.is_emoji(c)]

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', str(text))
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = text.replace('\\', '').replace('"','').replace(':','')
    text = re.sub(r'\bRT\b[:]? ?', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['hashtags'] = df['translated_text'].apply(extract_hashtags)
df['mentions'] = df['translated_text'].apply(extract_mentions)
df['urls'] = df['translated_text'].apply(extract_urls)
df['emojis'] = df['translated_text'].apply(extract_emojis)
df['clean_text'] = df['translated_text'].apply(clean_text)
df['hashtags_count'] = df['hashtags'].apply(len)
df['user_mentions_count'] = df['mentions'].apply(len)
df['urls_count'] = df['urls'].apply(len)
df['emojis_count'] = df['emojis'].apply(len)
df['num_posts_user'] = df.groupby('username')['post_id'].transform('count')


# Load BERT

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()


# Semantic GAT

class SemanticGAT(nn.Module):
    def __init__(self, hidden_size, out_size, num_edge_types=5):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.att = nn.Linear(hidden_size*2 + num_edge_types, 1)
        self.out_proj = nn.Linear(hidden_size, out_size)

    def forward(self, embeddings, adj_matrix):
        B, T, H = embeddings.shape
        h = self.W(embeddings)
        h_i = h.unsqueeze(2).expand(B, T, T, H)
        h_j = h.unsqueeze(1).expand(B, T, T, H)
        edge_feat = F.one_hot(adj_matrix.to(torch.int64), num_classes=5).float()
        concat = torch.cat([h_i, h_j, edge_feat], dim=-1)
        alpha = self.att(concat).squeeze(-1)
        alpha = alpha.masked_fill(adj_matrix==0, -1e9)
        alpha = F.softmax(alpha, dim=-1)
        context = torch.matmul(alpha, h)
        sentence_vec = context.mean(dim=1)
        out = self.out_proj(sentence_vec)
        return out

dep_att_layer = SemanticGAT(hidden_size=768, out_size=128).to(device)


# Dependency adjacency

def build_dep_adj(doc):
    seq_len = len(doc)
    adj = torch.zeros(seq_len, seq_len)
    for tok in doc:
        if tok.head.i != tok.i:
            adj[tok.i, tok.head.i] = 1
            adj[tok.head.i, tok.i] = 1
        if tok.dep_ == 'neg':
            adj[tok.i, tok.head.i] = adj[tok.head.i, tok.i] = 2
        if tok.dep_ in ['amod','advmod']:
            adj[tok.i, tok.head.i] = adj[tok.head.i, tok.i] = 3
        if tok.dep_ in ['tmod','npadvmod']:
            adj[tok.i, tok.head.i] = adj[tok.head.i, tok.i] = 4
    return adj

def build_dep_adj_bert_aligned(text, tokenizer, max_length=64):
    """
    Builds a dependency adjacency matrix aligned to BERT subword tokens.
    Uses word_ids() to map SpaCy token indices → BERT subword indices.
    """
    doc = nlp(text)
    spacy_tokens = [tok.text for tok in doc]
    
    # Tokenize with return_offsets_mapping for alignment
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    # word_ids() maps each BERT subword position → word index (None for [CLS]/[SEP])
    word_ids = encoded.word_ids(batch_index=0)  # list of len max_length
    
    bert_seq_len = max_length
    adj = torch.zeros(bert_seq_len, bert_seq_len)
    
    # Build SpaCy-level adjacency first (same as before)
    spacy_adj = {}
    for tok in doc:
        i, j = tok.i, tok.head.i
        if i == j:
            continue
        edge_type = 1  # default
        if tok.dep_ == 'neg':
            edge_type = 2
        elif tok.dep_ in ['amod', 'advmod']:
            edge_type = 3
        elif tok.dep_ in ['tmod', 'npadvmod']:
            edge_type = 4
        spacy_adj[(i, j)] = edge_type
        spacy_adj[(j, i)] = edge_type  # undirected
    
    # Map SpaCy word indices → all corresponding BERT subword indices
    word_to_bert = {}
    for bert_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_to_bert:
            word_to_bert[word_idx] = []
        word_to_bert[word_idx].append(bert_idx)
    
    # Propagate edges: for each SpaCy edge (i→j), connect ALL subwords of i to ALL subwords of j
    for (spacy_i, spacy_j), edge_type in spacy_adj.items():
        bert_i_positions = word_to_bert.get(spacy_i, [])
        bert_j_positions = word_to_bert.get(spacy_j, [])
        for bi in bert_i_positions:
            for bj in bert_j_positions:
                if bi < bert_seq_len and bj < bert_seq_len:
                    adj[bi, bj] = edge_type
                    adj[bj, bi] = edge_type
        
        # Also connect subwords of the same word to each other (intra-word edges)
        for positions in [bert_i_positions, bert_j_positions]:
            for bi in positions:
                for bj in positions:
                    if bi != bj and bi < bert_seq_len and bj < bert_seq_len:
                        adj[bi, bj] = 1  # same-word edge
    
    return adj, encoded


# Text embeddings + semantic vectors

batch_size = 16
texts = df['clean_text'].tolist()
all_global_embeddings, all_local_embeddings, semantic_vectors = [], [], []

for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
    batch_texts = texts[i:i+batch_size]
    
    # Build aligned adjacency matrices per sample
    batch_adjs = []
    batch_input_ids = []
    batch_attention_masks = []
    
    for text in batch_texts:
        adj, encoded = build_dep_adj_bert_aligned(text, tokenizer, max_length=64)
        batch_adjs.append(adj)
        batch_input_ids.append(encoded['input_ids'].squeeze(0))
        batch_attention_masks.append(encoded['attention_mask'].squeeze(0))
    
    # Stack into batch tensors
    input_ids = torch.stack(batch_input_ids).to(device)
    attention_mask = torch.stack(batch_attention_masks).to(device)
    batch_adj_tensor = torch.stack(batch_adjs).to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        cls_emb = last_hidden[:, 0, :]
        all_global_embeddings.extend(cls_emb.cpu().tolist())
        all_local_embeddings.extend([emb.cpu().tolist() for emb in last_hidden])
        
        dep_vec = dep_att_layer(last_hidden, batch_adj_tensor)
        semantic_vectors.extend(dep_vec.cpu().tolist())

df['global_embedding'] = all_global_embeddings
df['local_embeddings'] = all_local_embeddings
df['semantic_vector'] = semantic_vectors
df.to_pickle("Dataset/twitter/df_with_embeddings.pkl")
print(" Text embeddings + semantic vectors saved!")


# Image Dataset

img_folder = "Dataset/twitter/images_train"
image_transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

class TwitterDataset(Dataset):
    def __init__(self, df, img_folder, transform=None):
        self.df = df
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, f"{row['image_id']}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            image = Image.new("RGB",(224,224),(0,0,0))
        if self.transform:
            image = self.transform(image)
        return {"image": image, "semantic_vector": torch.tensor(row['semantic_vector'], dtype=torch.float)}

dataset = TwitterDataset(df, img_folder, transform=image_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


# Models

# ViT + CNN
vit_model_name = "google/vit-base-patch16-224-in21k"
vit_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)
vit_model = ViTModel.from_pretrained(vit_model_name).to(device).eval()
cnn_model = models.resnet50(pretrained=True)
cnn_model = nn.Sequential(*list(cnn_model.children())[:-2]).to(device).eval()

# YOLO
yolo_model = YOLO("yolov8n.pt")
obj_proj_layer = nn.Linear(2048,512).to(device)

# Vision Fusion
class VisionFusionModule(nn.Module):
    def __init__(self, vit_dim=768, cnn_dim=2048, obj_dim=512, fused_dim=1024):
        super().__init__()
        self.vit_proj = nn.Linear(vit_dim, fused_dim)
        self.cnn_proj = nn.Linear(cnn_dim, fused_dim)
        self.obj_proj = nn.Linear(obj_dim, fused_dim)
        self.fusion_gate = nn.Linear(fused_dim*3, fused_dim)
        self.activation = nn.ReLU()

    def forward(self, vit_emb, cnn_emb, obj_emb):
        vit = self.activation(self.vit_proj(vit_emb))
        cnn = self.activation(self.cnn_proj(cnn_emb))
        obj = self.activation(self.obj_proj(obj_emb))
        fused = self.activation(self.fusion_gate(torch.cat([vit, cnn, obj], dim=-1)))
        return fused

fusion_model = VisionFusionModule().to(device)


# Embeddings cache

cache_path = "Dataset/twitter/image_embeddings_cache.pkl"
if os.path.exists(cache_path):
    print("Loading cached image embeddings...")
    with open(cache_path, "rb") as f:
        cached_data = pickle.load(f)
    image_embeddings = cached_data['image_embeddings']
    text_embeddings = cached_data['text_embeddings']
else:
    print("No cache found. Computing embeddings...")
    image_embeddings, text_embeddings = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image embeddings"):
            images = batch['image'].to(device)
            semantic_vec = batch['semantic_vector'].to(device)
            pil_images = [T.ToPILImage()(img.cpu()) for img in images]

            # ViT
            vit_encoded = vit_extractor(images=pil_images, return_tensors="pt")['pixel_values'].to(device)
            vit_out = vit_model(vit_encoded).last_hidden_state[:,0,:]

            # CNN
            cnn_out = cnn_model(images).mean(dim=[2,3])

            # YOLO object embeddings
            batch_obj_feats = []
            all_crops, crop_batch_map = [], []
            for idx, img_pil in enumerate(pil_images):
                results = yolo_model(img_pil)
                boxes = results[0].boxes.xyxy.cpu() if len(results[0].boxes) > 0 else torch.empty((0,4))
                if len(boxes) == 0:
                    batch_obj_feats.append(torch.zeros(512, device=device))
                    continue
                for box in boxes:
                    x1,y1,x2,y2 = map(int, box)
                    obj_crop = img_pil.crop((x1,y1,x2,y2))
                    obj_tensor = image_transform(obj_crop)
                    all_crops.append(obj_tensor)
                    crop_batch_map.append(idx)

            if len(all_crops) > 0:
                crop_batch = torch.stack(all_crops).to(device)
                crop_feats = cnn_model(crop_batch).mean(dim=[2,3])
                crop_feats = obj_proj_layer(crop_feats)
                img_feat_dict = {i: [] for i in range(len(pil_images))}
                for feat, img_idx in zip(crop_feats, crop_batch_map):
                    img_feat_dict[img_idx].append(feat)
                for i in range(len(pil_images)):
                    if i in img_feat_dict and len(img_feat_dict[i]) > 0:
                        batch_obj_feats.append(torch.stack(img_feat_dict[i]).mean(dim=0))
                    elif len(batch_obj_feats) < i+1:
                        batch_obj_feats.append(torch.zeros(512, device=device))
            else:
                batch_obj_feats = [torch.zeros(512, device=device) for _ in range(len(pil_images))]

            obj_emb = torch.stack(batch_obj_feats)
            fused_img = fusion_model(vit_out, cnn_out, obj_emb)
            image_embeddings.append(fused_img.cpu())
            text_embeddings.append(semantic_vec.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    with open(cache_path, "wb") as f:
        pickle.dump({'image_embeddings': image_embeddings, 'text_embeddings': text_embeddings}, f)
    print(" Cached image embeddings saved!")


#  Derived features (numeric & categorical)

numeric_features = ['hashtags_count', 'user_mentions_count', 'urls_count', 'emojis_count', 'num_posts_user']
categorical_features = ['username']


# Temporal encoding (sin/cos)

def temporal_encoding(timestamps, period=24*60*60):
    sin_enc = np.sin(2 * np.pi * timestamps / period)
    cos_enc = np.cos(2 * np.pi * timestamps / period)
    return np.stack([sin_enc, cos_enc], axis=1)


# Preprocessing

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])
preprocessor.fit(df)
numeric_cat_array = preprocessor.transform(df)

timestamps = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
timestamps_np = timestamps.to_numpy(dtype='datetime64[s]')
epoch = np.datetime64('1970-01-01T00:00:00')
timestamps_unix = (timestamps_np - epoch).astype(np.float32)
temporal_array = temporal_encoding(timestamps_unix)

metadata_array = np.hstack([numeric_cat_array, temporal_array])
metadata_tensor = torch.tensor(metadata_array, dtype=torch.float32).to(device)


# Dense embedding projection

class MetadataEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.linear(x))

metadata_embed_model = MetadataEmbedding(input_dim=metadata_tensor.shape[1], embed_dim=128).to(device)
dense_embeddings = metadata_embed_model(metadata_tensor)


# Optional Sequence Modeling (per-user GRU)

user_groups = df.groupby('username')['post_id'].apply(list).to_dict()
gru_input_dim = dense_embeddings.shape[1]
gru_hidden_dim = 128
gru_model = nn.GRU(input_size=gru_input_dim, hidden_size=gru_hidden_dim, batch_first=True).to(device)
user_embedding_dict = {}

with torch.no_grad():
    for user, post_ids in user_groups.items():
        user_indices = df[df['post_id'].isin(post_ids)].index.tolist()
        user_sequence = dense_embeddings[user_indices].unsqueeze(0)
        _, h_n = gru_model(user_sequence)
        user_embedding_dict[user] = h_n.squeeze(0)

metadata_embedding_vector = torch.stack([user_embedding_dict[u] for u in df['username']], dim=0)


#  Save raw embeddings

torch.save(dense_embeddings.cpu(), "metadata_dense_embeddings.pt")
torch.save(metadata_embedding_vector.cpu(), "metadata_user_sequence_embeddings.pt")
print(" Dense embeddings shape:", dense_embeddings.shape)
print(" Metadata sequence embeddings shape:", metadata_embedding_vector.shape)


#  PREPROCESSING PIPELINE (BEFORE MODEL TRAINING)


class EmbeddingPreprocessor:
    """
    Implements the complete pipeline:
    1. Extract embeddings (already done)
    2. Normalize (L2)
    3. Dimensionality reduction per modality
    4. Align embeddings (CCA)
    5. Fuse embeddings
    6. Optional post-fusion reduction
    7. Cluster
    8. Evaluate
    """
    
    def __init__(self, 
                 pca_components_text=64,
                 pca_components_image=64,
                 pca_components_meta=64,
                 cca_components=64,
                 fusion_method='weighted',
                 post_fusion_method='pca',
                 post_fusion_dim=128,
                 cluster_method='hdbscan'):
        
        self.pca_components_text = pca_components_text
        self.pca_components_image = pca_components_image
        self.pca_components_meta = pca_components_meta
        self.cca_components = cca_components
        self.fusion_method = fusion_method
        self.post_fusion_method = post_fusion_method
        self.post_fusion_dim = post_fusion_dim
        self.cluster_method = cluster_method
        
        self.pca_text = None
        self.pca_image = None
        self.pca_meta = None
        self.cca_text_image = None
        self.post_fusion_reducer = None
        self.clusterer = None
        self.fusion_weights = None
        
    def l2_normalize(self, embeddings):
        """Normalize embeddings using L2 norm"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def fit_pca_per_modality(self, text_emb, image_emb, meta_emb):
        print("\nStep 3: Dimensionality reduction per modality (PCA)")

        # Fix metadata shape if it’s 3D (e.g., [N, 1, 128])
        if meta_emb.ndim == 3:
            meta_emb = meta_emb.squeeze(1)

        # PCA for text
        self.pca_text = PCA(n_components=self.pca_components_text)
        text_reduced = self.pca_text.fit_transform(text_emb)
        print(f"  Text: {text_emb.shape[1]} → {self.pca_components_text} (explained variance: {self.pca_text.explained_variance_ratio_.sum():.3f})")

        # PCA for image
        self.pca_image = PCA(n_components=self.pca_components_image)
        image_reduced = self.pca_image.fit_transform(image_emb)
        print(f"  Image: {image_emb.shape[1]} → {self.pca_components_image} (explained variance: {self.pca_image.explained_variance_ratio_.sum():.3f})")

        # PCA for meta
        self.pca_meta = PCA(n_components=self.pca_components_meta)
        meta_reduced = self.pca_meta.fit_transform(meta_emb)
        print(f"  Meta: {meta_emb.shape[1]} → {self.pca_components_meta} (explained variance: {self.pca_meta.explained_variance_ratio_.sum():.3f})")

        return text_reduced, image_reduced, meta_reduced
    
    def align_embeddings_cca(self, text_reduced, image_reduced):
        """Step 4: CCA removed — correlation was 0.063, adding noise not signal.
        Pass PCA-reduced embeddings directly as separate streams."""
        print(f"  Text: {text_reduced.shape}, Image: {image_reduced.shape}")
        return text_reduced, image_reduced
        
    def fuse_embeddings(self, text_aligned, image_aligned, meta_reduced):
        """Step 5: Fuse embeddings"""
        print(f"\n Step 5: Fuse embeddings (method: {self.fusion_method})")
        
        if self.fusion_method == 'concatenate':
            fused = np.concatenate([text_aligned, image_aligned, meta_reduced], axis=1)
            print(f"  Concatenated dimensions: {fused.shape[1]}")
            
        elif self.fusion_method == 'weighted':
            target_dim = min(text_aligned.shape[1], image_aligned.shape[1], meta_reduced.shape[1])
            
            text_proj = text_aligned[:, :target_dim]
            image_proj = image_aligned[:, :target_dim]
            meta_proj = meta_reduced[:, :target_dim]
            
            text_var = np.var(text_proj, axis=0).mean()
            image_var = np.var(image_proj, axis=0).mean()
            meta_var = np.var(meta_proj, axis=0).mean()
            
            total_var = text_var + image_var + meta_var
            w_text = text_var / total_var
            w_image = image_var / total_var
            w_meta = meta_var / total_var
            
            self.fusion_weights = np.array([w_text, w_image, w_meta])
            
            fused = (w_text * text_proj + 
                    w_image * image_proj + 
                    w_meta * meta_proj)
            
            print(f"  Fusion weights: text={w_text:.3f}, image={w_image:.3f}, meta={w_meta:.3f}")
            print(f"  Fused dimensions: {fused.shape[1]}")
        
        return fused
    
    def post_fusion_reduction(self, fused):
        """Step 6: Optional dimensionality reduction after fusion"""
        if self.post_fusion_method is None:
            print("\n⏩ Step 6: Skipping post-fusion reduction")
            return fused
        
        print(f"\n📊 Step 6: Post-fusion reduction ({self.post_fusion_method})")
        
        if self.post_fusion_method == 'pca':
            self.post_fusion_reducer = PCA(n_components=self.post_fusion_dim)
            reduced = self.post_fusion_reducer.fit_transform(fused)
            explained_var = self.post_fusion_reducer.explained_variance_ratio_.sum()
            print(f"  PCA: {fused.shape[1]} → {reduced.shape[1]} "
                  f"(explained variance: {explained_var:.3f})")
            
        elif self.post_fusion_method == 'umap':
            self.post_fusion_reducer = umap.UMAP(n_components=self.post_fusion_dim, 
                                                  random_state=42)
            reduced = self.post_fusion_reducer.fit_transform(fused)
            print(f"  UMAP: {fused.shape[1]} → {reduced.shape[1]}")
        
        return reduced
    
    def cluster_embeddings(self, embeddings):
        """Step 7: Cluster embeddings"""
        print(f"\n🔍 Step 7: Clustering ({self.cluster_method})")
        
        if self.cluster_method == 'hdbscan':
            self.clusterer = HDBSCAN(min_cluster_size=10, 
                                     min_samples=5,
                                     metric='euclidean')
            cluster_labels = self.clusterer.fit_predict(embeddings)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f"  HDBSCAN found {n_clusters} clusters ({n_noise} noise points)")
            
        elif self.cluster_method == 'spectral':
            n_clusters = 5
            self.clusterer = SpectralClustering(n_clusters=n_clusters, 
                                                random_state=42,
                                                affinity='nearest_neighbors')
            cluster_labels = self.clusterer.fit_predict(embeddings)
            print(f"  Spectral clustering: {n_clusters} clusters")
        
        return cluster_labels
    
    def evaluate_clustering(self, embeddings, cluster_labels):
        """Step 8: Evaluate clustering quality"""
        print("\n📈 Step 8: Evaluate clustering")
        
        if -1 in cluster_labels:
            mask = cluster_labels != -1
            if mask.sum() < 2:
                print("  ⚠️ Too few non-noise points to compute silhouette score")
                return None
            embeddings_clean = embeddings[mask]
            labels_clean = cluster_labels[mask]
        else:
            embeddings_clean = embeddings
            labels_clean = cluster_labels
        
        n_unique_labels = len(set(labels_clean))
        if n_unique_labels < 2:
            print("  ⚠️ Need at least 2 clusters to compute silhouette score")
            return None
        
        silhouette = silhouette_score(embeddings_clean, labels_clean)
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  (Range: [-1, 1], higher is better, >0.5 is good)")
        
        return silhouette
    
    def fit_transform(self, text_emb, image_emb, meta_emb):
        """Complete pipeline"""
        print("\n" + "="*70)
        print("MULTIMODAL EMBEDDING PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 2: Normalize
        print("\n Step 2: L2 Normalization")
        text_norm = self.l2_normalize(text_emb)
        image_norm = self.l2_normalize(image_emb)
        meta_norm = self.l2_normalize(meta_emb)
        print(f"  Text norm: mean={np.linalg.norm(text_norm, axis=1).mean():.3f}")
        print(f"  Image norm: mean={np.linalg.norm(image_norm, axis=1).mean():.3f}")
        print(f"  Meta norm: mean={np.linalg.norm(meta_norm, axis=1).mean():.3f}")
        
        # Step 3: Dimensionality reduction
        text_reduced, image_reduced, meta_reduced = self.fit_pca_per_modality(
            text_norm, image_norm, meta_norm
        )
        
        # Step 4: Alignment
        text_aligned, image_aligned = self.align_embeddings_cca(
            text_reduced, image_reduced
        )
        
        # Step 5: Fusion
        fused = self.fuse_embeddings(text_aligned, image_aligned, meta_reduced)
        
        # Step 6: Post-fusion reduction
        final_embeddings = self.post_fusion_reduction(fused)
        
        # Step 7: Clustering
        cluster_labels = self.cluster_embeddings(final_embeddings)
        
        # Step 8: Evaluation
        silhouette = self.evaluate_clustering(final_embeddings, cluster_labels)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
        return {
            'final_embeddings': final_embeddings,
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette,
            'text_normalized': text_norm,
            'image_normalized': image_norm,
            'meta_normalized': meta_norm,
            'text_reduced': text_reduced,
            'image_reduced': image_reduced,
            'meta_reduced': meta_reduced,
            'text_aligned': text_aligned,
            'image_aligned': image_aligned,
            'fused': fused
        }



#  APPLY PREPROCESSING PIPELINE (CRITICAL: BEFORE MODEL TRAINING)


print("\n" + "="*70)
print("APPLYING PREPROCESSING PIPELINE TO EMBEDDINGS")
print("="*70)

# Prepare embeddings
# Align all embeddings to the same rows as image_embeddings (11,844)
# image_embeddings was built from df rows that had valid images
n_img = image_embeddings.shape[0]

text_embeddings_array = np.array(semantic_vectors)[:n_img]
image_embeddings_array = image_embeddings.cpu().numpy()
meta_embeddings_array = metadata_embedding_vector.cpu().numpy()[:n_img]

# Also align df for downstream use (contradiction scores, labels etc.)
df = df.iloc[:n_img].reset_index(drop=True)

print(f"Aligned to {n_img} rows (image cache size)")

print(f"\nInput shapes:")
print(f"  Text embeddings: {text_embeddings_array.shape}")
print(f"  Image embeddings: {image_embeddings_array.shape}")
print(f"  Metadata embeddings: {meta_embeddings_array.shape}")

# Initialize preprocessor
embedding_preprocessor = EmbeddingPreprocessor(
    pca_components_text=64,
    pca_components_image=64,
    pca_components_meta=32,
    cca_components=64,        # unused now but harmless
    fusion_method='concatenate',
    post_fusion_method=None,  # no post-fusion collapse
    cluster_method='hdbscan'
)

# Run the complete pipeline
preprocessing_results = embedding_preprocessor.fit_transform(
    text_embeddings_array,
    image_embeddings_array,
    meta_embeddings_array
)

torch.save(torch.tensor(preprocessing_results['text_aligned']), "Dataset/twitter/text_aligned.pt")
torch.save(torch.tensor(preprocessing_results['image_aligned']), "Dataset/twitter/image_aligned.pt")
torch.save(torch.tensor(preprocessing_results['meta_reduced']), "Dataset/twitter/meta_reduced.pt")

# Save cluster labels for analysis
df['cluster_label'] = preprocessing_results['cluster_labels']


# 🔥 USE PREPROCESSED EMBEDDINGS FOR MODEL TRAINING


# Convert preprocessed embeddings back to tensors
preprocessed_text_embeddings = torch.tensor(
    preprocessing_results['text_aligned'], 
    dtype=torch.float32
)
preprocessed_image_embeddings = torch.tensor(
    preprocessing_results['image_aligned'], 
    dtype=torch.float32
)
preprocessed_meta_embeddings = torch.tensor(
    preprocessing_results['meta_reduced'], 
    dtype=torch.float32
)

print("\n" + "="*70)
print("PREPROCESSED EMBEDDINGS READY FOR TRAINING")
print("="*70)
print(f"  Text (aligned): {preprocessed_text_embeddings.shape}")
print(f"  Image (aligned): {preprocessed_image_embeddings.shape}")
print(f"  Meta (reduced): {preprocessed_meta_embeddings.shape}")

# Save preprocessor for inference
preprocessor_path = "Dataset/twitter/embedding_preprocessor.pkl"
with open(preprocessor_path, 'wb') as f:
    pickle.dump(embedding_preprocessor, f)
print(f"\n Preprocessor saved to {preprocessor_path}")


# Inner Fusion (Text + Image) - NOW USING PREPROCESSED EMBEDDINGS

class InnerFusionModule(nn.Module):
    def __init__(self, text_dim=64, image_dim=64, fused_dim=1024):  # Updated dims
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.cross_gate = nn.Linear(fused_dim*2, fused_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, text_emb, image_emb):
        t = self.activation(self.text_proj(text_emb))
        v = self.activation(self.image_proj(image_emb))
        fused = self.activation(self.cross_gate(torch.cat([t,v], dim=-1)))
        fused = self.norm(fused)
        return fused

cross_fusion = InnerFusionModule(
    text_dim=preprocessed_text_embeddings.shape[1],
    image_dim=preprocessed_image_embeddings.shape[1]
).to(device)

# Cross Verifier
class CrossVerifier(nn.Module):
    def __init__(self, fused_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(fused_dim, fused_dim//2)
        self.fc2 = nn.Linear(fused_dim//2, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fused_emb):
        x = self.activation(self.fc1(fused_emb))
        x = self.sigmoid(self.fc2(x))
        return x

cross_verifier_model = CrossVerifier().to(device)


# Prepare CrossVerifier training WITH PREPROCESSED EMBEDDINGS

# Label-aware negative sampling
fake_indices = torch.tensor([i for i, label in enumerate(df['label']) 
                              if label in [1, 'fake', 'Fake']])
real_indices = torch.tensor([i for i, label in enumerate(df['label']) 
                              if label in [0, 'real', 'Real']])

# Positive pairs: text + its own image
pos_text   = preprocessed_text_embeddings
pos_image  = preprocessed_image_embeddings
pos_labels = torch.ones(len(pos_text))

# Hard negatives: fake text + real image (deception-indicative mismatch)
n_hard = min(len(fake_indices), len(real_indices))
hard_fake_idx   = fake_indices[torch.randperm(len(fake_indices))[:n_hard]]
hard_real_idx   = real_indices[torch.randperm(len(real_indices))[:n_hard]]
hard_neg_text   = preprocessed_text_embeddings[hard_fake_idx]
hard_neg_image  = preprocessed_image_embeddings[hard_real_idx]
hard_neg_labels = torch.zeros(n_hard)

# Easy negatives: random shuffled pairs
easy_perm       = torch.randperm(len(preprocessed_image_embeddings))
easy_neg_text   = preprocessed_text_embeddings
easy_neg_image  = preprocessed_image_embeddings[easy_perm]
easy_neg_labels = torch.zeros(len(easy_neg_text))

# Combine
train_text_emb  = torch.cat([pos_text,  hard_neg_text,  easy_neg_text],  dim=0)
train_image_emb = torch.cat([pos_image, hard_neg_image, easy_neg_image], dim=0)
train_labels    = torch.cat([pos_labels, hard_neg_labels, easy_neg_labels], dim=0)

print(f"Training pairs: {len(train_labels)} total")
print(f"  Positive (matched):        {len(pos_labels)}")
print(f"  Hard negative (fake+real): {n_hard}")
print(f"  Easy negative (random):    {len(easy_neg_labels)}")

train_dataset = torch.utils.data.TensorDataset(train_text_emb, train_image_emb, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(
    list(cross_fusion.parameters()) + list(cross_verifier_model.parameters()), 
    lr=1e-4
)
criterion = nn.BCELoss()
epochs = 5

print("\n" + "="*70)
print("TRAINING CROSS VERIFIER WITH PREPROCESSED EMBEDDINGS")
print("="*70)

for epoch in range(epochs):
    cross_fusion.train()
    cross_verifier_model.train()
    epoch_loss = 0
    for text_emb, img_emb, labels in train_loader:
        text_emb, img_emb, labels = text_emb.to(device), img_emb.to(device), labels.to(device).float()
        optimizer.zero_grad()
        fused_cross = cross_fusion(text_emb, img_emb)
        preds = cross_verifier_model(fused_cross).squeeze()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


# Generate contradiction scores WITH PREPROCESSED EMBEDDINGS

cross_fusion.eval()
cross_verifier_model.eval()
with torch.no_grad():
    all_fused = cross_fusion(
        preprocessed_text_embeddings.to(device), 
        preprocessed_image_embeddings.to(device)
    )
    contradiction_scores = cross_verifier_model(all_fused).cpu().numpy()

df['contradiction_score'] = contradiction_scores
df.to_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
print("\n Contradiction scores generated with preprocessed embeddings!")


# ADAPTIVE MULTIMODAL FUSION LAYER - NOW USING PREPROCESSED EMBEDDINGS


import torch
import torch.nn as nn
import torch.nn.functional as F


# STEP 1: INPUT PROJECTION - UPDATED DIMENSIONS


class Step1_InputProjection(nn.Module):
    def __init__(self, d_text=64, d_image=64, d_meta=64, d_common=256):  # Updated
        super().__init__()
        self.d_common = d_common
        self.proj_text = nn.Sequential(
            nn.Linear(d_text, d_text),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_text, d_common),
            nn.ReLU()
        )
        self.proj_image = nn.Sequential(
            nn.Linear(d_image, d_image),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_image, d_common),
            nn.ReLU()
        )
        self.proj_meta = nn.Sequential(
            nn.Linear(d_meta, d_meta),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_meta, d_common),
            nn.ReLU()
        )
        self.norm_text = nn.LayerNorm(d_common)
        self.norm_image = nn.LayerNorm(d_common)
        self.norm_meta = nn.LayerNorm(d_common)

    def forward(self, h_text, h_image, h_meta):
        z_text = self.norm_text(self.proj_text(h_text))
        z_image = self.norm_image(self.proj_image(h_image))
        z_meta = self.norm_meta(self.proj_meta(h_meta))
        return z_text, z_image, z_meta



# STEP 2: CROSS-MODAL ATTENTION


class Step2_CrossModalAttention(nn.Module):
    def __init__(self, d_common):
        super().__init__()
        self.scale = d_common ** 0.5
        
        self.query_text = nn.Linear(d_common, d_common)
        self.key_image = nn.Linear(d_common, d_common)
        self.value_image = nn.Linear(d_common, d_common)
        
        self.query_image = nn.Linear(d_common, d_common)
        self.key_text = nn.Linear(d_common, d_common)
        self.value_text = nn.Linear(d_common, d_common)
        
        self.proj_meta = nn.Linear(d_common, d_common)
        
        self.text_residual_weight = nn.Parameter(torch.tensor(0.7))
        self.image_residual_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, z_text, z_image, z_meta):
        Q_t = self.query_text(z_text)    # [B, d]
        K_i = self.key_image(z_image)    # [B, d]
        V_i = self.value_image(z_image)  # [B, d]
        
        # Per-sample dot product — scalar attention weight per post
        # (Q_t * K_i) is element-wise, sum gives one scalar per sample
        attn_score_ti = (Q_t * K_i).sum(dim=-1, keepdim=True) / self.scale  # [B, 1]
        attn_weight_ti = torch.sigmoid(attn_score_ti)  # [B, 1]
        text_attended = attn_weight_ti * V_i            # [B, d]
        
        z_text_attn = (self.text_residual_weight * z_text + 
                       (1 - self.text_residual_weight) * text_attended)

        Q_i = self.query_image(z_image)  # [B, d]
        K_t = self.key_text(z_text)      # [B, d]
        V_t = self.value_text(z_text)    # [B, d]
        
        attn_score_it = (Q_i * K_t).sum(dim=-1, keepdim=True) / self.scale  # [B, 1]
        attn_weight_it = torch.sigmoid(attn_score_it)  # [B, 1]
        image_attended = attn_weight_it * V_t           # [B, d]
        
        z_image_attn = (self.image_residual_weight * z_image + 
                        (1 - self.image_residual_weight) * image_attended)

        z_meta_attn = self.proj_meta(z_meta)

        # Average attention weights for interpretability
        avg_attn = (attn_weight_ti + attn_weight_it) / 2  # [B, 1]

        return z_text_attn, z_image_attn, z_meta_attn, avg_attn


# STEP 3: MISMATCH VECTOR


class Step3_MismatchVector(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(d_common, d_common),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_common, d_common // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_common // 2, d_common)
        )

    def forward(self, z_text_attn, z_image_attn):
        diff = z_text_attn - z_image_attn
        v_mismatch = self.mismatch_encoder(diff)
        return v_mismatch



# STEP 4: PATTERN LEARNER


class ModalityPatternLearner(nn.Module):
    def __init__(self, d_common=256, hidden_dim=128):
        super().__init__()
        
        self.text_scorer = nn.Sequential(
            nn.Linear(d_common, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.image_scorer = nn.Sequential(
            nn.Linear(d_common, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.meta_scorer = nn.Sequential(
            nn.Linear(d_common, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.text_conf = nn.Sequential(
            nn.Linear(d_common, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.image_conf = nn.Sequential(
            nn.Linear(d_common, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.meta_conf = nn.Sequential(
            nn.Linear(d_common, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z_text, z_image, z_meta):
        suspicion_scores = torch.cat([
            self.text_scorer(z_text),
            self.image_scorer(z_image),
            self.meta_scorer(z_meta)
        ], dim=1)
        
        pattern_confidence = torch.cat([
            self.text_conf(z_text),
            self.image_conf(z_image),
            self.meta_conf(z_meta)
        ], dim=1)
        
        return suspicion_scores, pattern_confidence



# STEP 4b: MISMATCH ANALYZER


class ModalityMismatchAnalyzer(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        self.alignment_scorer = nn.Sequential(
            nn.Linear(d_common * 2, d_common),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_common, 1),
            nn.Sigmoid()
        )

    def forward(self, z_text, z_image, v_mismatch):
        ti_concat = torch.cat([z_text, z_image], dim=1)
        alignment_score = self.alignment_scorer(ti_concat)
        
        mismatch_mag = torch.norm(v_mismatch, dim=1, keepdim=True)
        mag_norm = (mismatch_mag - mismatch_mag.min()) / (mismatch_mag.max() - mismatch_mag.min() + 1e-8)
        is_contradictory = (1 - alignment_score) * mag_norm
        
        return alignment_score, mismatch_mag, is_contradictory



# STEP 4c: ADAPTIVE MODALITY WEIGHTING


class AdaptiveModalityWeighting(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        
        self.weight_gen = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        self.norm = nn.LayerNorm(d_common)

    def forward(self, z_text, z_image, z_meta, suspicion_scores, pattern_confidence, is_contradictory):
        contradiction_exp = is_contradictory.expand(-1, 3)
        weight_input = torch.cat([suspicion_scores, pattern_confidence, contradiction_exp], dim=1)
        
        modality_weights = self.weight_gen(weight_input)
        
        z_fused = (
            modality_weights[:, 0:1] * z_text +
            modality_weights[:, 1:2] * z_image +
            modality_weights[:, 2:3] * z_meta
        )
        z_fused = self.norm(z_fused)
        
        return z_fused, modality_weights



# STEP 5: CLASSIFICATION HEAD WITH UNCERTAINTY


class ClassificationHeadWithUncertainty(nn.Module):
    def __init__(self, d_in=256, d_hidden=256):
        super().__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.BatchNorm1d(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(d_hidden // 2, 1)
        self.uncertainty_head = nn.Linear(d_hidden // 2, 1)

    def forward(self, z_fused):
        features = self.feature_processor(z_fused)
        logit = self.classifier(features)
        uncertainty_score = torch.sigmoid(self.uncertainty_head(features))
        return logit, uncertainty_score



# COMPLETE MODEL - WITH PREPROCESSED EMBEDDING DIMENSIONS


class AdaptiveMultimodalFakeNewsDetector(nn.Module):
    def __init__(self, d_text=64, d_image=64, d_meta=64, d_common=256):  # Updated
        super().__init__()
        self.step1 = Step1_InputProjection(d_text, d_image, d_meta, d_common)
        self.step2 = Step2_CrossModalAttention(d_common)
        self.step3 = Step3_MismatchVector(d_common)
        self.pattern_learner = ModalityPatternLearner(d_common)
        self.mismatch_analyzer = ModalityMismatchAnalyzer(d_common)
        self.adaptive_weighting = AdaptiveModalityWeighting(d_common)
        self.classifier = ClassificationHeadWithUncertainty(d_common)

    def forward(self, h_text, h_image, h_meta, return_intermediates=False):
        z_text, z_image, z_meta = self.step1(h_text, h_image, h_meta)
        z_text_attn, z_image_attn, z_meta_attn, attn_weights = self.step2(z_text, z_image, z_meta)
        v_mismatch = self.step3(z_text_attn, z_image_attn)
        suspicion_scores, pattern_confidence = self.pattern_learner(z_text, z_image, z_meta)
        alignment_score, mismatch_mag, is_contradictory = self.mismatch_analyzer(
            z_text, z_image, v_mismatch
        )
        z_fused, modality_weights = self.adaptive_weighting(
            z_text_attn, z_image_attn, z_meta_attn,
            suspicion_scores, pattern_confidence, is_contradictory
        )
        logit, uncertainty_score = self.classifier(z_fused)

        outputs = {
            "logit": logit,
            "uncertainty_score": uncertainty_score,
            "modality_weights": modality_weights,
            "suspicion_scores": suspicion_scores,
            "pattern_confidence": pattern_confidence,
            "alignment_score": alignment_score,
            "mismatch_magnitude": mismatch_mag,
            "is_contradictory": is_contradictory,
            "attn_weights": attn_weights,
            "z_fused": z_fused,
            "v_mismatch": v_mismatch
        }

        if return_intermediates:
            return logit, outputs
        return logit



# LOSSES


class ContrastiveMismatchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_mismatch, labels):
        v_norm = F.normalize(v_mismatch, p=2, dim=1)
        sim_matrix = torch.mm(v_norm, v_norm.t()) / self.temperature
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(mask * log_prob).sum(dim=1).mean()
        return loss



# HELPER FUNCTIONS


def unpack_batch(batch, device):
    """Extract batch tensors safely"""
    if isinstance(batch, dict):
        h_text = batch['text'].to(device)
        h_image = batch['image'].to(device)
        h_meta = batch['metadata'].to(device)
        labels = batch['label'].to(device).float()
    elif isinstance(batch, (list, tuple)):
        h_text = batch[0].to(device)
        h_image = batch[1].to(device)
        h_meta = batch[2].to(device)
        labels = batch[3].to(device).float()
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    return h_text, h_image, h_meta, labels


def train_step(model, batch, criterion, mismatch_loss_fn, optimizer, 
               lambda_contrast=0.1, lambda_suspicion=0.05, lambda_alignment=0.05):
    """Single training step"""
    h_text, h_image, h_meta, labels = unpack_batch(batch, device)
    
    optimizer.zero_grad()
    logits, intermediates = model(h_text, h_image, h_meta, return_intermediates=True)
    
    v_mismatch = intermediates['v_mismatch']
    suspicion_scores = intermediates['suspicion_scores']
    alignment_score = intermediates['alignment_score']
    
    loss_bce = criterion(logits.squeeze(), labels)
    loss_contrast = mismatch_loss_fn(v_mismatch, labels.long())
    loss_suspicion = criterion(suspicion_scores.mean(dim=1), labels)
    loss_alignment = criterion(alignment_score.squeeze(), 1.0 - labels)
    
    loss = (loss_bce + 
            lambda_contrast * loss_contrast + 
            lambda_suspicion * loss_suspicion + 
            lambda_alignment * loss_alignment)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {
        'loss_total': loss.item(),
        'loss_bce': loss_bce.item(),
        'loss_contrast': loss_contrast.item(),
        'loss_suspicion': loss_suspicion.item(),
        'loss_alignment': loss_alignment.item()
    }



# TEST WITH PREPROCESSED EMBEDDINGS


print("\n" + "="*70)
print("INITIALIZING MODEL WITH PREPROCESSED EMBEDDING DIMENSIONS")
print("="*70)

model = AdaptiveMultimodalFakeNewsDetector(
    d_text=preprocessed_text_embeddings.shape[1],
    d_image=preprocessed_image_embeddings.shape[1],
    d_meta=preprocessed_meta_embeddings.shape[1],
    d_common=256
).to(device)

print(" Model created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create test batch with preprocessed embeddings
batch_size_test = 8
batch = {
    'text': preprocessed_text_embeddings[:batch_size_test].to(device),
    'image': preprocessed_image_embeddings[:batch_size_test].to(device),
    'metadata': preprocessed_meta_embeddings[:batch_size_test].to(device),
    'label': torch.randint(0, 2, (batch_size_test,)).float().to(device)
}

print("\nRunning forward pass with preprocessed embeddings...")
with torch.no_grad():
    logits, outputs = model(batch['text'], batch['image'], batch['metadata'], return_intermediates=True)
    print(f" Output logits shape: {logits.shape}")
    print(f" Modality weights: {outputs['modality_weights'][0]}")

print("\nCluster analysis:")
cluster_counts = df['cluster_label'].value_counts().sort_index()
print(cluster_counts)
if preprocessing_results['silhouette_score']:
    print(f"\nSilhouette score: {preprocessing_results['silhouette_score']:.3f}")

print("\n ALL PREPROCESSING AND INTEGRATION COMPLETE!")
print(" Models are now using normalized, aligned, and dimensionality-reduced embeddings!")
print(" This should significantly improve training performance and convergence!")