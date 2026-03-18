"""
complete_ablation_analysis.py
Full ablation study - copy this entire file and run it with your data loaders.
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import os
import pickle
import ast
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


print("Loading data...")

CACHE_PATH = "Dataset/twitter/image_embeddings_cache.pkl"
DF_PATH = "Dataset/twitter/df_train_translated.csv"
METADATA_EMB_PT = "metadata_user_sequence_embeddings.pt"
LABEL_COL = "label"

# Load dataframe
df = pd.read_csv(DF_PATH)
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()

label_map = {
    'fake': 0, 'f': 0, '0': 0, 'false': 0, 'not_real': 0,
    'real': 1, 'r': 1, '1': 1, 'true': 1
}

df['_mapped_label'] = df[LABEL_COL].map(label_map)
valid_mask = ~df['_mapped_label'].isna()
df = df.loc[valid_mask].reset_index(drop=True)
df[LABEL_COL] = df['_mapped_label'].astype(int)
df.drop(columns=['_mapped_label'], inplace=True)

# Load embeddings
with open(CACHE_PATH, "rb") as f:
    cached = pickle.load(f)

image_embeddings = cached.get('image_embeddings')
text_embeddings = cached.get('text_embeddings')

if isinstance(image_embeddings, torch.Tensor):
    image_embeddings = image_embeddings.cpu().numpy()
if isinstance(text_embeddings, torch.Tensor):
    text_embeddings = text_embeddings.cpu().numpy()

# Truncate to match df
n_samples = len(df)
image_embeddings = image_embeddings[:n_samples]
text_embeddings = text_embeddings[:n_samples]

# Load metadata
if os.path.exists(METADATA_EMB_PT):
    metadata_user_seq = torch.load(METADATA_EMB_PT)
    if isinstance(metadata_user_seq, torch.Tensor):
        arr = metadata_user_seq.cpu().numpy()
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        metadata_embeddings = arr
    else:
        metadata_embeddings = np.array(metadata_user_seq)
    metadata_embeddings = metadata_embeddings[:n_samples]
else:
    print("Metadata embedding not found, using zeros")
    metadata_embeddings = np.zeros((n_samples, 128), dtype=np.float32)

# Parse semantic vectors from df
if 'semantic_vector' in df.columns:
    sem_list = []
    for v in df['semantic_vector'].values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            sem_list.append(np.zeros(text_embeddings.shape[1], dtype=np.float32))
            continue
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
            except:
                try:
                    parsed = np.fromstring(v.strip('[]'), sep=',')
                except:
                    parsed = np.zeros(text_embeddings.shape[1], dtype=np.float32)
            sem_list.append(np.array(parsed, dtype=np.float32))
        elif isinstance(v, (list, tuple, np.ndarray)):
            sem_list.append(np.array(v, dtype=np.float32))
        else:
            sem_list.append(np.zeros(text_embeddings.shape[1], dtype=np.float32))
    
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
    semantic_vectors = text_embeddings.copy()

y = df[LABEL_COL].astype(int).values

# Create loaders
X_text = semantic_vectors.astype(np.float32)
X_image = image_embeddings.astype(np.float32)
X_meta = metadata_embeddings.astype(np.float32)

train_idx, test_idx = train_test_split(np.arange(n_samples), test_size=0.2, random_state=SEED, stratify=y)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED, stratify=y[train_idx])

def make_loader(idxs, batch_size=64, shuffle=True):
    t_text = torch.tensor(X_text[idxs]).to(torch.float32)
    t_img  = torch.tensor(X_image[idxs]).to(torch.float32)
    t_meta = torch.tensor(X_meta[idxs]).to(torch.float32)
    t_lab  = torch.tensor(y[idxs]).to(torch.float32)
    ds = TensorDataset(t_text, t_img, t_meta, t_lab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(train_idx, shuffle=True)
val_loader = make_loader(val_idx, shuffle=False)
test_loader = make_loader(test_idx, shuffle=False)

print(f"Data loaded: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

#Baseline models

class TextOnlyBaseline(nn.Module):
    def __init__(self, d_text=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_text, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, h_text, h_image, h_meta):
        return self.classifier(h_text)

class ImageOnlyBaseline(nn.Module):
    def __init__(self, d_image=1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_image, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, h_text, h_image, h_meta):
        return self.classifier(h_image)

class MetadataOnlyBaseline(nn.Module):
    def __init__(self, d_meta=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_meta, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, h_text, h_image, h_meta):
        return self.classifier(h_meta)

class NaiveConcatenation(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128):
        super().__init__()
        total_dim = d_text + d_image + d_meta
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, h_text, h_image, h_meta):
        combined = torch.cat([h_text, h_image, h_meta], dim=1)
        return self.classifier(combined)

class SimpleProjectionAverage(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.proj_text = nn.Linear(d_text, d_common)
        self.proj_image = nn.Linear(d_image, d_common)
        self.proj_meta = nn.Linear(d_meta, d_common)
        self.classifier = nn.Sequential(
            nn.Linear(d_common, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, h_text, h_image, h_meta):
        z_text = self.proj_text(h_text)
        z_image = self.proj_image(h_image)
        z_meta = self.proj_meta(h_meta)
        z_avg = (z_text + z_image + z_meta) / 3.0
        return self.classifier(z_avg)

class ProjectionConcat(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.proj_text = nn.Linear(d_text, d_common)
        self.proj_image = nn.Linear(d_image, d_common)
        self.proj_meta = nn.Linear(d_meta, d_common)
        self.classifier = nn.Sequential(
            nn.Linear(d_common * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, h_text, h_image, h_meta):
        z_text = self.proj_text(h_text)
        z_image = self.proj_image(h_image)
        z_meta = self.proj_meta(h_meta)
        combined = torch.cat([z_text, z_image, z_meta], dim=1)
        return self.classifier(combined)

class ProjectionAttention(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.proj_text = nn.Linear(d_text, d_common)
        self.proj_image = nn.Linear(d_image, d_common)
        self.proj_meta = nn.Linear(d_meta, d_common)
        
        self.attn_t_to_i = nn.MultiheadAttention(d_common, 8, batch_first=True)
        self.attn_i_to_t = nn.MultiheadAttention(d_common, 8, batch_first=True)
        self.norm = nn.LayerNorm(d_common)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_common * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, h_text, h_image, h_meta):
        z_text = self.proj_text(h_text)
        z_image = self.proj_image(h_image)
        z_meta = self.proj_meta(h_meta)
        
        z_text_seq = z_text.unsqueeze(1)
        z_image_seq = z_image.unsqueeze(1)
        
        z_text_attn, _ = self.attn_t_to_i(z_text_seq, z_image_seq, z_image_seq)
        z_text_attn = self.norm(z_text_attn.squeeze(1) + z_text)
        
        z_image_attn, _ = self.attn_i_to_t(z_image_seq, z_text_seq, z_text_seq)
        z_image_attn = self.norm(z_image_attn.squeeze(1) + z_image)
        
        combined = torch.cat([z_text_attn, z_image_attn, z_meta], dim=1)
        return self.classifier(combined)

class ProjectionAttentionMismatch(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128, d_common=256):
        super().__init__()
        self.proj_text = nn.Linear(d_text, d_common)
        self.proj_image = nn.Linear(d_image, d_common)
        self.proj_meta = nn.Linear(d_meta, d_common)
        
        self.attn_t_to_i = nn.MultiheadAttention(d_common, 8, batch_first=True)
        self.attn_i_to_t = nn.MultiheadAttention(d_common, 8, batch_first=True)
        self.norm = nn.LayerNorm(d_common)
        
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(d_common, d_common),
            nn.ReLU(),
            nn.Linear(d_common, d_common)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_common * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, h_text, h_image, h_meta):
        z_text = self.proj_text(h_text)
        z_image = self.proj_image(h_image)
        z_meta = self.proj_meta(h_meta)
        
        z_text_seq = z_text.unsqueeze(1)
        z_image_seq = z_image.unsqueeze(1)
        
        z_text_attn, _ = self.attn_t_to_i(z_text_seq, z_image_seq, z_image_seq)
        z_text_attn = self.norm(z_text_attn.squeeze(1) + z_text)
        
        z_image_attn, _ = self.attn_i_to_t(z_image_seq, z_text_seq, z_text_seq)
        z_image_attn = self.norm(z_image_attn.squeeze(1) + z_image)
        
        diff = z_text_attn - z_image_attn
        v_mismatch = self.norm(self.mismatch_encoder(diff))
        
        combined = torch.cat([z_text_attn, z_image_attn, z_meta, v_mismatch], dim=1)
        return self.classifier(combined)

#Evaluation

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for t_text, t_img, t_meta, t_lab in test_loader:
            t_text = t_text.to(device)
            t_img = t_img.to(device)
            t_meta = t_meta.to(device)
            
            logits = model(t_text, t_img, t_meta)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds.tolist())
            all_scores.extend(probs.tolist())
            all_labels.extend(t_lab.numpy().astype(int).tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = float('nan')
    
    return {'accuracy': acc, 'f1': f1, 'auc': auc}

 #Run ablation

def run_ablation(train_loader, val_loader, test_loader, device=DEVICE):
    
    models_to_test = {
        '1. Text Only': TextOnlyBaseline(d_text=128),
        '2. Image Only': ImageOnlyBaseline(d_image=1024),
        '3. Metadata Only': MetadataOnlyBaseline(d_meta=128),
        '4. Naive Concat': NaiveConcatenation(d_text=128, d_image=1024, d_meta=128),
        '5. Proj + Avg': SimpleProjectionAverage(d_text=128, d_image=1024, d_meta=128, d_common=256),
        '6. Proj + Concat': ProjectionConcat(d_text=128, d_image=1024, d_meta=128, d_common=256),
        '7. Proj + Attention': ProjectionAttention(d_text=128, d_image=1024, d_meta=128, d_common=256),
        '8. Proj + Attn + Mismatch': ProjectionAttentionMismatch(d_text=128, d_image=1024, d_meta=128, d_common=256),
    }
    
    results = {}
    criterion = nn.BCEWithLogitsLoss()
    
    for name, model in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        best_val_f1 = 0
        patience = 3
        patience_counter = 0
        
        for epoch in range(20):
            model.train()
            for t_text, t_img, t_meta, t_lab in train_loader:
                t_text = t_text.to(device)
                t_img = t_img.to(device)
                t_meta = t_meta.to(device)
                t_lab = t_lab.to(device).float()
                
                optimizer.zero_grad()
                logits = model(t_text, t_img, t_meta)
                loss = criterion(logits.squeeze(), t_lab)
                loss.backward()
                optimizer.step()
            
            val_metrics = evaluate_model(model, val_loader, device)
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: val_f1={val_metrics['f1']:.4f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        test_metrics = evaluate_model(model, test_loader, device)
        results[name] = test_metrics
        
        print(f"TEST - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    return results


# print summary


def print_summary(results):
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    print(f"{'Model':<40} {'Accuracy':<15} {'F1':<15} {'AUC':<15}")
    print("-"*100)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<40} {metrics['accuracy']:.4f}        {metrics['f1']:.4f}        {metrics['auc']:.4f}")
    
    print("="*100)
    
    print("\nKEY FINDINGS:")
    
    text_acc = results.get('1. Text Only', {}).get('accuracy', 0)
    image_acc = results.get('2. Image Only', {}).get('accuracy', 0)
    meta_acc = results.get('3. Metadata Only', {}).get('accuracy', 0)
    
    print(f"\n1. Single Modality Performance:")
    print(f"   Text: {text_acc:.4f}")
    print(f"   Image: {image_acc:.4f}")
    print(f"   Metadata: {meta_acc:.4f}")
    print(f"   Best single: {max(text_acc, image_acc, meta_acc):.4f}")
    
    best_single_acc = max(text_acc, image_acc, meta_acc)
    naive_acc = results.get('4. Naive Concat', {}).get('accuracy', 0)
    proj_concat = results.get('6. Proj + Concat', {}).get('accuracy', 0)
    proj_attn = results.get('7. Proj + Attention', {}).get('accuracy', 0)
    proj_attn_mis = results.get('8. Proj + Attn + Mismatch', {}).get('accuracy', 0)
    
    print(f"\n2. Fusion Benefit (Naive Concat vs Best Single):")
    print(f"   Improvement: {naive_acc - best_single_acc:.4f} ({((naive_acc - best_single_acc)/best_single_acc*100):.2f}%)")
    
    print(f"\n3. Projection Benefit:")
    print(f"   Proj+Concat vs Naive Concat: {proj_concat - naive_acc:.4f}")
    
    print(f"\n4. Attention Benefit:")
    print(f"   Proj+Attention vs Proj+Concat: {proj_attn - proj_concat:.4f}")
    
    print(f"\n5. Mismatch Vector Benefit:")
    print(f"   Proj+Attn+Mismatch vs Proj+Attention: {proj_attn_mis - proj_attn:.4f}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    print("\nRunning comprehensive ablation study...\n")
    results = run_ablation(train_loader, val_loader, test_loader)
    print_summary(results)
    
    # Save results
    np.savez("ablation_results.npz", **{k: np.array([v['accuracy'], v['f1'], v['auc']]) for k, v in results.items()})
    print("\nResults saved to ablation_results.npz")