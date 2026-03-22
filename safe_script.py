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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pickle 
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cluster import HDBSCAN, SpectralBiclustering
from sklearn.metrics import silhouette_score
import umap


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load SpaCy

nlp = spacy.load("en_core_web_sm")


# Load Dataset
df = pd.read_csv("Dataset/twitter/df_train_translated.csv")


# Text preprocessing
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
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

def align_adj_to_bert(adj, bert_seq_len):
    padded = torch.zeros(bert_seq_len, bert_seq_len)
    seq_len = adj.shape[0]
    if seq_len >= bert_seq_len:
        padded = adj[:bert_seq_len, :bert_seq_len]
    else:
        padded[:seq_len, :seq_len] = adj
    return padded

# Text embeddings + semantic vectors

batch_size = 16
texts = df['clean_text'].tolist()
all_global_embeddings, all_local_embeddings, semantic_vectors = [], [], []

for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
    batch_texts = texts[i:i+batch_size]
    encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    encoded = {k:v.to(device) for k,v in encoded.items()}
    seq_len = encoded['input_ids'].shape[1]

    with torch.no_grad():
        outputs = bert_model(**encoded)
        last_hidden = outputs.last_hidden_state
        cls_emb = last_hidden[:,0,:]
        all_global_embeddings.extend(cls_emb.cpu().tolist())
        all_local_embeddings.extend([emb.cpu().tolist() for emb in last_hidden])

    batch_adj = torch.zeros(len(batch_texts), seq_len, seq_len, device=device)
    for b, text in enumerate(batch_texts):
        doc = nlp(text)
        adj = build_dep_adj(doc)
        batch_adj[b] = align_adj_to_bert(adj, seq_len)

    with torch.no_grad():
        dep_vec = dep_att_layer(last_hidden, batch_adj)
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

# Inner Fusion (Text + Image)
class InnerFusionModule(nn.Module):
    def __init__(self, text_dim=128, image_dim=1024, fused_dim=1024):
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

cross_fusion = InnerFusionModule().to(device)

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
    print("Cached image embeddings saved!")


# Prepare CrossVerifier training
pos_pairs = torch.arange(len(image_embeddings))
pos_labels = torch.ones(len(pos_pairs))
neg_pairs = torch.randperm(len(image_embeddings))
neg_labels = torch.zeros(len(neg_pairs))
train_text_emb = torch.cat([text_embeddings, text_embeddings], dim=0)
train_image_emb = torch.cat([image_embeddings, image_embeddings[neg_pairs]], dim=0)
train_labels = torch.cat([pos_labels, neg_labels], dim=0)
train_dataset = torch.utils.data.TensorDataset(train_text_emb, train_image_emb, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(list(cross_fusion.parameters()) + list(cross_verifier_model.parameters()), lr=1e-4)
criterion = nn.BCELoss()
epochs = 5

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


# Generate contradiction scores
cross_fusion.eval()
cross_verifier_model.eval()
with torch.no_grad():
    all_fused = cross_fusion(text_embeddings.to(device), image_embeddings.to(device))
    contradiction_scores = cross_verifier_model(all_fused).cpu().numpy()

df['contradiction_score'] = contradiction_scores
df.to_pickle("Dataset/twitter/df_with_contradiction_scores.pkl")
print(" Contradiction scores generated and saved!")


#  Derived features (numeric & categorical)
numeric_features = ['hashtags_count', 'user_mentions_count', 'urls_count', 'emojis_count', 'num_posts_user']
categorical_features = ['username']


# Temporal encoding (sin/cos)
def temporal_encoding(timestamps, period=24*60*60):
    sin_enc = np.sin(2 * np.pi * timestamps / period)
    cos_enc = np.cos(2 * np.pi * timestamps / period)
    return np.stack([sin_enc, cos_enc], axis=1)


#  Preprocessing
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


#  Dense embedding projection

class MetadataEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.linear(x))

metadata_embed_model = MetadataEmbedding(input_dim=metadata_tensor.shape[1], embed_dim=128).to(device)
dense_embeddings = metadata_embed_model(metadata_tensor)


#  Optional Sequence Modeling (per-user GRU)

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


#  Save embeddings

torch.save(dense_embeddings.cpu(), "metadata_dense_embeddings.pt")
torch.save(metadata_embedding_vector.cpu(), "metadata_user_sequence_embeddings.pt")
print(" Dense embeddings shape:", dense_embeddings.shape)
print(" Metadata sequence embeddings shape:", metadata_embedding_vector.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# INPUT PROJECTION


class Step1_InputProjection(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
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



#  2: CROSS-MODAL ATTENTION


class Step2_CrossModalAttention(nn.Module):
    def __init__(self, d_common, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = d_common ** 0.5
        
        # Text pathway (keeps text mostly intact)
        self.query_text = nn.Linear(d_common, d_common)
        self.key_image = nn.Linear(d_common, d_common)
        self.value_image = nn.Linear(d_common, d_common)
        
        # Image pathway (keeps image mostly intact)
        self.query_image = nn.Linear(d_common, d_common)
        self.key_text = nn.Linear(d_common, d_common)
        self.value_text = nn.Linear(d_common, d_common)
        
        # Metadata (pass through)
        self.proj_meta = nn.Linear(d_common, d_common)
        
        # RESIDUAL CONNECTIONS (preserve original signal)
        self.text_residual_weight = nn.Parameter(torch.tensor(0.7))  # Keep 70% original text
        self.image_residual_weight = nn.Parameter(torch.tensor(0.7))  # Keep 70% original image

    def forward(self, z_text, z_image, z_meta):
        # Text attends to image
        Q_t = self.query_text(z_text)
        K_i = self.key_image(z_image)
        V_i = self.value_image(z_image)
        attn_scores_ti = torch.matmul(Q_t, K_i.T) / self.scale
        attn_weights_ti = F.softmax(attn_scores_ti, dim=-1)
        text_attended = torch.matmul(attn_weights_ti, V_i)
        
        # RESIDUAL: Keep original text + weighted attention
        z_text_attn = self.text_residual_weight * z_text + (1 - self.text_residual_weight) * text_attended

        # Image attends to text
        Q_i = self.query_image(z_image)
        K_t = self.key_text(z_text)
        V_t = self.value_text(z_text)
        attn_scores_it = torch.matmul(Q_i, K_t.T) / self.scale
        attn_weights_it = F.softmax(attn_scores_it, dim=-1)
        image_attended = torch.matmul(attn_weights_it, V_t)
        
        # RESIDUAL: Keep original image + weighted attention
        z_image_attn = self.image_residual_weight * z_image + (1 - self.image_residual_weight) * image_attended

        z_meta_attn = self.proj_meta(z_meta)

        return z_text_attn, z_image_attn, z_meta_attn, (attn_weights_ti + attn_weights_it) / 2
'''

**Key changes:**
1. Added residual connections (keep 70% original, 30% attended)
2. Made residual weights learnable parameters
3. This preserves variance while still allowing cross-modal interaction


'''
'''
After attention:
  z_text_attn: ~0.7-0.8  (instead of 0.001621)
  z_image_attn: ~0.7-0.8 (instead of 0.001562)
'''

#  MISMATCH VECTOR 


class Step3_MismatchVector(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        #Removed LayerNorm to allow mismatch discrimination
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



# PATTERN LEARNER 


class ModalityPatternLearner(nn.Module):
    def __init__(self, d_common=256, hidden_dim=128):
        super().__init__()
        
        # FIXED: All scorers output [B, 1] consistently
        self.text_scorer = nn.Sequential(
            nn.Linear(d_common, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # FIXED: Changed from 8 to 1
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
        
        # Confidence scorers
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
        ], dim=1)  # Result: [B, 3]
        
        pattern_confidence = torch.cat([
            self.text_conf(z_text),        
            self.image_conf(z_image),      
            self.meta_conf(z_meta)         
        ], dim=1)  # Result: [B, 3]
        
        return suspicion_scores, pattern_confidence


#  4b: MISMATCH ANALYZER


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


#  4c: ADAPTIVE MODALITY WEIGHTING (FIXED - Correct input size + Softmax)


class AdaptiveModalityWeighting(nn.Module):
    def __init__(self, d_common=256):
        super().__init__()
        
        # FIXED: Input size corrected from 16 to 9
        # suspicion_scores: [B, 3]
        # pattern_confidence: [B, 3]
        # contradiction_expanded: [B, 3]
        # Total: [B, 9]
        
        self.weight_gen = nn.Sequential(
            nn.Linear(9, 64),  # FIXED: Corrected input size
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)  # FIXED: Added Softmax for positive weights
        )
        
        self.norm = nn.LayerNorm(d_common)

    def forward(self, z_text, z_image, z_meta, suspicion_scores, pattern_confidence, is_contradictory):
        contradiction_exp = is_contradictory.expand(-1, 3)  # [B, 3]
        weight_input = torch.cat([suspicion_scores, pattern_confidence, contradiction_exp], dim=1)  # [B, 9]
        
        modality_weights = self.weight_gen(weight_input)  # [B, 3], sums to 1
        
        # Weighted fusion
        z_fused = (
            modality_weights[:, 0:1] * z_text +
            modality_weights[:, 1:2] * z_image +
            modality_weights[:, 2:3] * z_meta
        )
        z_fused = self.norm(z_fused)
        
        return z_fused, modality_weights



#  5: CLASSIFICATION HEAD WITH UNCERTAINTY


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



# COMPLETE MODEL


class AdaptiveMultimodalFakeNewsDetector(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.step1 = Step1_InputProjection(d_text, d_image, d_meta, d_common)
        self.step2 = Step2_CrossModalAttention(d_common)
        self.step3 = Step3_MismatchVector(d_common)
        self.pattern_learner = ModalityPatternLearner(d_common)
        self.mismatch_analyzer = ModalityMismatchAnalyzer(d_common)
        self.adaptive_weighting = AdaptiveModalityWeighting(d_common)
        self.classifier = ClassificationHeadWithUncertainty(d_common)

    def forward(self, h_text, h_image, h_meta, return_intermediates=False):
        # Step 1: Project to common space
        z_text, z_image, z_meta = self.step1(h_text, h_image, h_meta)
        
        # Step 2: Cross-modal attention
        z_text_attn, z_image_attn, z_meta_attn, attn_weights = self.step2(z_text, z_image, z_meta)
        
        # Step 3: Mismatch
        v_mismatch = self.step3(z_text_attn, z_image_attn)
        
        # Step 4: Pattern learning
        suspicion_scores, pattern_confidence = self.pattern_learner(z_text, z_image, z_meta)
        
        # Step 4b: Mismatch analysis
        alignment_score, mismatch_mag, is_contradictory = self.mismatch_analyzer(
            z_text, z_image, v_mismatch
        )
        
        # Step 4c: Adaptive weighting
        z_fused, modality_weights = self.adaptive_weighting(
            z_text_attn, z_image_attn, z_meta_attn,
            suspicion_scores, pattern_confidence, is_contradictory
        )
        
        # Step 5: Classification
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


def check_gradients(model, batch, criterion, device):
    """Check gradient flow through critical layers"""
    model.train()
    h_text, h_image, h_meta, labels = unpack_batch(batch, device)
    
    logits, intermediates = model(h_text, h_image, h_meta, return_intermediates=True)
    loss = criterion(logits.squeeze(), labels)
    loss.backward()
    
    print("\n" + "="*70)
    print("GRADIENT FLOW DIAGNOSTICS")
    print("="*70)
    
    problem_found = False
    for name, param in model.named_parameters():
        if 'weight_gen' in name or 'scorer' in name or 'mismatch_encoder' in name:
            if param.grad is None:
                print(f"ERROR: NO GRADIENT: {name}")
                problem_found = True
            elif param.grad.abs().mean() < 1e-6:
                print(f"WARNING: TINY GRADIENT: {name} = {param.grad.abs().mean():.2e}")
                problem_found = True
            else:
                print(f"OK: {name} = {param.grad.abs().mean():.2e}")
    
    print("="*70)
    if problem_found:
        print("WARNING: Some layers have gradient issues!")
    else:
        print("OK: All critical layers receiving gradients.")
    print("="*70 + "\n")
    
    return not problem_found


def diagnose_fusion_layer(model, batch):
    """Run full diagnostics"""
    model.eval()
    with torch.no_grad():
        h_text = batch['text'].to(device)
        h_image = batch['image'].to(device)
        h_meta = batch['metadata'].to(device)
        logits, intermediates = model(h_text, h_image, h_meta, return_intermediates=True)

        print("\n" + "="*70)
        print("FUSION LAYER DIAGNOSTICS")
        print("="*70)

        print("\n1. DIMENSIONS:")
        for name in ["z_fused", "v_mismatch", "modality_weights"]:
            if name in intermediates:
                print(f"  {name}: {tuple(intermediates[name].shape)}")
        print(f"  Final logits: {tuple(logits.shape)}")

        if "modality_weights" in intermediates:
            w = intermediates["modality_weights"]
            print("\n2. ADAPTIVE WEIGHTS (should be positive, sum to 1):")
            print(f"  text={w[:,0].mean():.3f}±{w[:,0].std():.3f}, image={w[:,1].mean():.3f}±{w[:,1].std():.3f}, meta={w[:,2].mean():.3f}±{w[:,2].std():.3f}")
            print(f"  Sum per sample: {w.sum(dim=1).mean():.3f} (should be ~1.0)")
            if (w < 0).any():
                print("  ERROR: Negative weights detected!")
            if (w.std(dim=0) < 0.05).any():
                print("  WARNING: Low weight variance (not adapting)")

        if "v_mismatch" in intermediates:
            v_mismatch = intermediates["v_mismatch"]
            v_mag = torch.norm(v_mismatch, dim=1)
            print("\n3. MISMATCH VECTOR (should vary across samples):")
            print(f"  Mean={v_mag.mean():.4f}, Std={v_mag.std():.4f}, Range=[{v_mag.min():.4f}, {v_mag.max():.4f}]")
            if v_mag.std() < 0.01:
                print("  ERROR: Mismatch not discriminating (too uniform)!")

        if "suspicion_scores" in intermediates and "pattern_confidence" in intermediates:
            s = intermediates["suspicion_scores"]
            c = intermediates["pattern_confidence"]
            print("\n4. MODALITY SUSPICION & CONFIDENCE:")
            for i, name in enumerate(["Text", "Image", "Meta"]):
                print(f"  {name}: suspicion={s[:,i].mean():.3f}, confidence={c[:,i].mean():.3f}")

        if "is_contradictory" in intermediates and "alignment_score" in intermediates:
            contradiction = intermediates["is_contradictory"]
            align = intermediates["alignment_score"]
            print("\n5. CONTRADICTION SIGNALS:")
            print(f"  Alignment={align.mean():.3f}, Contradiction={contradiction.mean():.3f}")
            if contradiction.mean() > 0.6:
                print("  WARNING: Very high contradiction")

        print("\n" + "="*70)



# TEST


if __name__ == "__main__":
    print("Initializing model...")
    model = AdaptiveMultimodalFakeNewsDetector(
        d_text=128,
        d_image=1024,
        d_meta=128,
        d_common=256
    ).to(device)
    
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch = {
        'text': torch.randn(8, 128).to(device),
        'image': torch.randn(8, 1024).to(device),
        'metadata': torch.randn(8, 128).to(device),
        'label': torch.randint(0, 2, (8,)).float().to(device)
    }

    print("\nVariance Diagnostic:")
    with torch.no_grad():
        text_var = batch['text'].var(dim=0).mean().item()
        image_var = batch['image'].var(dim=0).mean().item()
        meta_var = batch['metadata'].var(dim=0).mean().item()
        
        print(f"\nInput variance:")
        print(f"  Text: {text_var:.6f}")
        print(f"  Image: {image_var:.6f}")
        print(f"  Meta: {meta_var:.6f}")
        
        # After projection
        z_text, z_image, z_meta = model.step1(batch['text'], batch['image'], batch['metadata'])
        
        print(f"\nProjected variance:")
        print(f"  z_text: {z_text.var(dim=0).mean().item():.6f}")
        print(f"  z_image: {z_image.var(dim=0).mean().item():.6f}")
        print(f"  z_meta: {z_meta.var(dim=0).mean().item():.6f}")
        
        # After attention
        z_text_attn, z_image_attn, z_meta_attn, _ = model.step2(z_text, z_image, z_meta)
        
        print(f"\nAfter attention:")
        print(f"  z_text_attn: {z_text_attn.var(dim=0).mean().item():.6f}")
        print(f"  z_image_attn: {z_image_attn.var(dim=0).mean().item():.6f}")
        print(f"  z_meta_attn: {z_meta_attn.var(dim=0).mean().item():.6f}")
    
    print("\nRunning full diagnostics...")
    diagnose_fusion_layer(model, batch)
    
    print("\nChecking gradient flow...")
    criterion = nn.BCEWithLogitsLoss()
    check_gradients(model, batch, criterion, device)
    
    print("\nAll tests completed!")
    
    
    
    
