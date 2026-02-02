import os
import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
# Image_scene_vector.py
# ---------------------
# Settings
# ---------------------
IMAGE_FOLDER = "Dataset/twitter/images_train"
OUTPUT_CSV = "Dataset/twitter/scene_emotions_vad_proj.csv"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

EMOTIONS = [
    "amusement", "awe", "contentment", "excitement",
    "anger", "disgust", "fear", "sadness", "calm", "tense"
]

EMOTION_VAD = {
    "amusement":     [0.85, 0.7, 0.75],
    "awe":           [0.7, 0.8, 0.65],
    "contentment":   [0.9, 0.3, 0.8],
    "excitement":    [0.8, 0.9, 0.7],
    "anger":         [0.1, 0.9, 0.8],
    "disgust":       [0.2, 0.6, 0.4],
    "fear":          [0.15, 0.85, 0.3],
    "sadness":       [0.2, 0.2, 0.3],
    "calm":          [0.8, 0.2, 0.7],
    "tense":         [0.3, 0.8, 0.6]
}

# ---------------------
# Load CLIP
# ---------------------
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()
print(f"⚡ Using device: {DEVICE}")
print(" CLIP model loaded")

# ---------------------
# VAD projector
# ---------------------
class VADProjector(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=512):  # align with CLIP feature dim
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, vad_scores):
        return self.projector(vad_scores)

vad_projector = VADProjector(input_dim=3, hidden_dim=128, output_dim=512).to(DEVICE)

# ---------------------
# Functions
# ---------------------
def extract_scene_emotion_clip(img_path, emotion_labels=EMOTIONS):
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f" Failed to load {img_path}: {e}")
        return None, None

    text_inputs = clip.tokenize([f"a photo that feels {label}" for label in emotion_labels]).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)
        probs = similarity.softmax(dim=0).cpu().numpy()

    emotion_dict = {emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))}
    return emotion_dict, image_features.cpu().numpy().flatten()

def probs_to_vad(emotion_probs, vad_mapping=EMOTION_VAD):
    valence = sum(emotion_probs[emo] * vad_mapping[emo][0] for emo in emotion_probs)
    arousal = sum(emotion_probs[emo] * vad_mapping[emo][1] for emo in emotion_probs)
    dominance = sum(emotion_probs[emo] * vad_mapping[emo][2] for emo in emotion_probs)
    return torch.tensor([valence, arousal, dominance], dtype=torch.float32).to(DEVICE)

# ---------------------
# Prepare training data for projector
# ---------------------
vad_tensors = []
clip_features = []

for img_file in tqdm(os.listdir(IMAGE_FOLDER), desc="Preparing training data"):
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    emotions, embedding = extract_scene_emotion_clip(img_path)
    if emotions is None:
        continue

    vad_scores = probs_to_vad(emotions)  # [3]
    vad_tensors.append(vad_scores)
    clip_features.append(torch.tensor(embedding, dtype=torch.float32).to(DEVICE))

# ---------------------
# Train VAD projector against CLIP embeddings
# ---------------------
optimizer = torch.optim.Adam(vad_projector.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()
vad_projector.train()
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for vad, target_feat in zip(vad_tensors, clip_features):
        optimizer.zero_grad()
        proj = vad_projector(vad.unsqueeze(0))
        loss = loss_fn(proj, target_feat.unsqueeze(0))  # map VAD -> CLIP embedding
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(vad_tensors):.6f}")

torch.save(vad_projector.state_dict(), "vad_projector.pth")
vad_projector.eval()
print("✅ VAD projector trained and saved")

# ---------------------
# Generate final embeddings
# ---------------------
results = []

for img_file in tqdm(os.listdir(IMAGE_FOLDER), desc="Generating final VAD embeddings"):
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    emotions, embedding = extract_scene_emotion_clip(img_path)
    if emotions is None:
        continue

    vad_scores = probs_to_vad(emotions).unsqueeze(0)
    with torch.no_grad():
        vad_embedding = vad_projector(vad_scores).cpu().numpy().flatten()

    vad_scores_flat = vad_scores.squeeze(0)

    row = {
        "image": img_file,
        "valence": float(vad_scores_flat[0]),
        "arousal": float(vad_scores_flat[1]),
        "dominance": float(vad_scores_flat[2]),
        "vad_embedding": ",".join(map(str, vad_embedding)),
        "clip_image_embedding": ",".join(map(str, embedding))
    }
    results.append(row)

# ---------------------
# Save CSV
# ---------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Scene VAD projections saved to {OUTPUT_CSV}")
