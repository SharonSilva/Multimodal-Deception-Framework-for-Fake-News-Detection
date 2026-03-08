"""
affective_embedding.py
----------------------

Fuses text, face, and scene VAD projections into a unified affective embedding
for multimodal fake news / deception detection.

Rules:
- Every post has text and scene VAD
- Some posts may have no face VAD (no face detected in image)
- post_id mapping ensures alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# -------------------------------------------------------
# 🔹 1. Emotion Fusion Layer — combines variable-dim VAD sources
# -------------------------------------------------------
class EmotionFusionLayer(nn.Module):
    """
    Fuses VAD embeddings from text, face, and scene into a unified affective embedding.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.Tanh()

    def forward(self, vad_text, vad_face, vad_scene):
        fused = torch.cat([vad_text, vad_face, vad_scene], dim=-1)
        fused = self.activation(self.norm1(self.fc1(fused)))
        affective_embedding = self.activation(self.norm2(self.fc2(fused)))
        return F.normalize(affective_embedding, dim=-1)

# -------------------------------------------------------
# 🔹 2. Utility: Align embeddings by post_id
# -------------------------------------------------------
def align_embeddings_by_post(df_embeddings, post_id_col, embedding_col, post_order, zero_fill_dim=None):
    """
    Align embeddings to posts using post_id mapping.
    If a post_id is missing, fill with zeros of the given dimension.

    Args:
        df_embeddings: DataFrame with columns [post_id_col, embedding_col]
        post_id_col: name of column for post_id
        embedding_col: name of column containing VAD embeddings
        post_order: list of post_ids in desired order
        zero_fill_dim: int, if embedding missing, fill with zeros of this size

    Returns:
        torch.Tensor of embeddings aligned to post_order
    """
    emb_dict = {pid: np.array(emb) for pid, emb in zip(df_embeddings[post_id_col], df_embeddings[embedding_col])}

    aligned_embeddings = []
    for pid in post_order:
        if pid in emb_dict:
            aligned_embeddings.append(emb_dict[pid])
        else:
            if zero_fill_dim is None:
                raise ValueError(f"Post {pid} missing embedding and zero_fill_dim is not provided")
            aligned_embeddings.append(np.zeros(zero_fill_dim, dtype=np.float32))

    return torch.tensor(np.stack(aligned_embeddings), dtype=torch.float32)

# -------------------------------------------------------
# 🔹 3. Affective Embedding Generator
# -------------------------------------------------------
class AffectiveEmbeddingGenerator:
    """
    Loads precomputed VAD projections (text, face, scene),
    aligns them per post, fuses them using EmotionFusionLayer,
    and outputs affective embeddings.
    """
    def __init__(self, text_vad_path, face_vad_path, scene_vad_path,
                 post_to_image_path, device="cpu"):
        self.device = device

        # Load post-to-image mapping
        df_post_map = pd.read_csv(post_to_image_path)  # must contain ['post_id','image_id']
        self.post_order = df_post_map['post_id'].tolist()

        # ---------------- Text embeddings ----------------
        self.vad_text = torch.load(text_vad_path).float()
        if len(self.vad_text) > len(self.post_order):
            self.vad_text = self.vad_text[:len(self.post_order)]

        # ---------------- Face embeddings ----------------
        # ---------------- Face embeddings (optional) ----------------
        try:
            df_face = pd.read_pickle(face_vad_path)
            df_face['image_filename'] = df_face['pth'].apply(lambda x: x.split('/')[-1])
            df_face = df_face.merge(df_post_map, left_on='image_filename', right_on='image_id', how='left')
            face_dim = len(df_face['image_vad_embedding'].iloc[0])
            self.vad_face = align_embeddings_by_post(
                df_face,
                post_id_col='post_id',
                embedding_col='image_vad_embedding',
                post_order=self.post_order,
                zero_fill_dim=face_dim
            )
            print(f"✅ Face VAD loaded: {self.vad_face.shape}")
        except Exception as e:
            print(f"⚠️  Face VAD unavailable ({e}) — using zeros")
            face_dim = 64
            self.vad_face = torch.zeros(len(self.post_order), face_dim)

        # ---------------- Scene embeddings ----------------
        df_scene = pd.read_csv(scene_vad_path)  # contains ['image','vad_embedding']

        # Convert string to array if needed
        if df_scene['vad_embedding'].dtype == object:
            df_scene['vad_embedding'] = df_scene['vad_embedding'].apply(lambda x: np.fromstring(x, sep=","))

        # Infer scene dimension from first row of CSV (before merging)
        scene_dim = len(df_scene['vad_embedding'].iloc[0])

        # Strip .jpg from image column to match image_id format
        df_scene['image'] = df_scene['image'].str.replace('.jpg', '', regex=False)

        # Merge with post mapping: image -> image_id -> post_id
        df_scene = df_scene.merge(df_post_map, left_on='image', right_on='image_id', how='left')

        # Convert post_id to int for consistency
        df_scene['post_id'] = df_scene['post_id'].fillna('__missing__')

        # Keep only valid post_ids
        df_scene_valid = df_scene[df_scene['post_id'] != '__missing__']

        # Align embeddings, zero-fill if missing
        self.vad_scene = align_embeddings_by_post(
            df_scene_valid,
            post_id_col='post_id',
            embedding_col='vad_embedding',
            post_order=self.post_order,
            zero_fill_dim=scene_dim
        )
        # Ensure same device
        n = len(self.vad_text)
        self.vad_face = self.vad_face[:n]
        self.vad_scene = self.vad_scene[:n]
        
        print(f"Aligned shapes — Text: {self.vad_text.shape}, Face: {self.vad_face.shape}, Scene: {self.vad_scene.shape}")

        # Ensure same device
        self.vad_text = self.vad_text.to(device)
        self.vad_face = self.vad_face.to(device)
        self.vad_scene = self.vad_scene.to(device)

        # Initialize fusion model
        input_dim = self.vad_text.shape[1] + self.vad_face.shape[1] + self.vad_scene.shape[1]
        self.model = EmotionFusionLayer(input_dim=input_dim).to(device)

    def generate(self, save_path=None):
        """Generate affective embeddings and optionally save to disk"""
        with torch.no_grad():
            affective_embedding = self.model(self.vad_text, self.vad_face, self.vad_scene)

        if save_path:
            np.save(save_path, affective_embedding.cpu().numpy())
            print(f"✅ Affective embeddings saved to {save_path}")

        return affective_embedding
    
    
    

# -------------------------------------------------------
# 🔹 4. Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    generator = AffectiveEmbeddingGenerator(
        text_vad_path="Dataset/twitter/text_vad_embedding.pt",
        face_vad_path="Dataset/affectnet/df_with_image_vad_embedding.pkl",
        scene_vad_path="Dataset/twitter/scene_emotions_vad_proj.csv",
        post_to_image_path="Dataset/twitter/df_train_translated.csv",
        device="cpu"
    )

    affective_embedding = generator.generate(
        save_path="Dataset/affectnet/affective_embedding.npy"
    )
    print("Affective embedding shape:", affective_embedding.shape)
    print("Sample:", affective_embedding[:5])
    
    
# import numpy as np

# # Load the saved embeddings
# affective_embeddings = np.load("Dataset/affectnet/affective_embedding.npy")

# print("Shape of embeddings:", affective_embeddings.shape)


# # Sum absolute values across each row (each post)
# zero_mask = np.sum(np.abs(affective_embeddings), axis=1) == 0

# # Count how many embeddings are zero
# num_zero_embeddings = np.sum(zero_mask)
# print(f"Number of posts with all-zero affective embeddings: {num_zero_embeddings} / {affective_embeddings.shape[0]}")