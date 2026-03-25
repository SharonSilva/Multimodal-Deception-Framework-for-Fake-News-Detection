from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import emoji
from tqdm import tqdm
import pandas as pd 

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_pickle("Dataset/twitter/df_with_embeddings.pkl")

emotion_model_name = "nateraw/bert-base-uncased-emotion"
tokenizer_emotion = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(device)
emotion_model.eval()


canonical_emotions = ["fear", "anger", "joy", "sadness", "neutral"]
canonical_indices = {em: i for i, em in enumerate(canonical_emotions)}

goemotions_labels = emotion_model.config.id2label
label_mapping = {
    "anger": "anger", "disgust": "anger",
    "fear": "fear", "joy": "joy",
    "sadness": "sadness", "neutral": "neutral",
    "surprise": "joy", "love": "joy",
    "optimism": "joy", "pessimism": "sadness"
}

vad_mapping = {
    "fear": np.array([0.2, 0.8, 0.3]),
    "anger": np.array([0.2, 0.8, 0.6]),
    "joy": np.array([0.9, 0.7, 0.8]),
    "sadness": np.array([0.1, 0.3, 0.4]),
    "neutral": np.array([0.5, 0.5, 0.5])
}

def map_goemotion_to_canonical(go_probs):
    canonical_vec = np.zeros(len(canonical_emotions), dtype=np.float32)
    for idx, label in goemotions_labels.items():
        canonical_label = label_mapping.get(label, "neutral")
        canonical_vec[canonical_indices[canonical_label]] += go_probs[idx]
    return canonical_vec / (canonical_vec.sum() + 1e-8)

def probs_to_vad(canonical_probs):
    vad = np.zeros(3, dtype=np.float32)
    for i, em in enumerate(canonical_emotions):
        vad += canonical_probs[i] * vad_mapping[em]
    return vad


# Emoji and Lexicon mappings

emoji_vad_map = {
    "😂": [0.95, 0.8, 0.7], "🤣": [0.95, 0.9, 0.8],
    "😊": [0.9, 0.6, 0.7], "😍": [0.9, 0.7, 0.8],
    "😢": [0.2, 0.5, 0.4], "😭": [0.1, 0.7, 0.3],
    "😡": [0.1, 0.8, 0.6], "😠": [0.2, 0.7, 0.5],
    "😨": [0.2, 0.9, 0.3], "😱": [0.15, 0.9, 0.3],
    "😐": [0.5, 0.4, 0.5], "😶": [0.5, 0.3, 0.5],
    "😴": [0.3, 0.2, 0.4], "🤔": [0.5, 0.5, 0.5],
    "❤️": [0.9, 0.7, 0.8], "💔": [0.2, 0.6, 0.4],
    "😎": [0.8, 0.5, 0.7], "🙄": [0.4, 0.4, 0.5],
    "😉": [0.8, 0.6, 0.7], "😞": [0.2, 0.4, 0.3],
    "😤": [0.3, 0.8, 0.6]
}

lexicon_vad = {
    "happy": [0.9, 0.7, 0.8],
    "sad": [0.2, 0.4, 0.4],
    "angry": [0.1, 0.8, 0.6],
    "fear": [0.2, 0.8, 0.3],
    "neutral": [0.5, 0.5, 0.5],
    "love": [0.9, 0.7, 0.8],
    "bored": [0.3, 0.3, 0.4],
    "tired": [0.3, 0.2, 0.4],
}


# Hybrid Emoji-Lexicon VAD computation

def get_vad_from_emoji(emj, lexicon):
    desc = emoji.demojize(emj)
    words = [w.strip(':') for w in desc.split('_')]
    vad_values = [lexicon.get(word, [0.5, 0.5, 0.5]) for word in words]
    return np.array(np.mean(vad_values, axis=0), dtype=np.float32)

def get_vad_from_text_with_emojis(text, lexicon, emoji_vad_map):
    vad_vectors = []
    for token in text.split():
        if token in emoji_vad_map:
            vad_vectors.append(np.array(emoji_vad_map[token], dtype=np.float32))
        elif any(ch in emoji.EMOJI_DATA for ch in token):
            vad_vectors.append(get_vad_from_emoji(token, lexicon))
        else:
            vad_vectors.append(np.array(lexicon.get(token.lower(), [0.5, 0.5, 0.5]), dtype=np.float32))
    if vad_vectors:
        return np.mean(vad_vectors, axis=0)
    return np.array([0.5, 0.5, 0.5], dtype=np.float32)


# Merge text + emoji list

def merge_text_emojis(text, emojis_list):
    emoji_text = ' '.join([emoji.demojize(e) for e in emojis_list])
    return f"{text} {emoji_text}".strip()


# Hybrid compute function

def compute_text_vad(texts, emojis_list=None, batch_size=32, fusion_weight=0.7):
    all_probs, all_vad = [], []

    if emojis_list is None:
        emojis_list = [[] for _ in texts]

    merged_texts = [merge_text_emojis(t, e) for t, e in zip(texts, emojis_list)]

    for i in tqdm(range(0, len(merged_texts), batch_size), desc="Computing Text VAD"):
        batch_texts = merged_texts[i:i + batch_size]
        encoded = tokenizer_emotion(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = emotion_model(**encoded).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        for text, emoji_list, p in zip(batch_texts, emojis_list[i:i + batch_size], probs):
            # Model-based VAD
            canonical_probs = map_goemotion_to_canonical(p)
            vad_model = probs_to_vad(canonical_probs)

            # Lexicon + Emoji-based VAD (as NumPy array)
            vad_lexemoji = get_vad_from_text_with_emojis(text, lexicon_vad, emoji_vad_map)

            # Weighted fusion
            vad_fused = fusion_weight * vad_model + (1 - fusion_weight) * vad_lexemoji

            all_probs.append(canonical_probs)
            all_vad.append(vad_fused)

    return np.array(all_probs, dtype=np.float32), np.array(all_vad, dtype=np.float32)


# Example usage

# df = pd.read_pickle("Dataset/twitter/cleaned_texts.pkl")  # must have columns 'clean_text' and 'emojis'
texts = df['clean_text'].tolist()
emojis_list = df['emojis'].tolist()

text_probs, text_vad = compute_text_vad(texts, emojis_list, batch_size=32)

df['text_emotion_probs'] = list(text_probs)
df['text_vad'] = list(text_vad)
df['text_valence'] = df['text_vad'].apply(lambda x: float(x[0]))
df['text_arousal'] = df['text_vad'].apply(lambda x: float(x[1]))
df['text_dominance'] = df['text_vad'].apply(lambda x: float(x[2]))

df.to_pickle("Dataset/twitter/df_with_text_emotions_vad.pkl")
print("✅ Text VAD saved to df_with_text_emotions_vad.pkl")


# VAD → Embedding projection (TRAINED)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class VADProjector(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, vad_scores):
        embedding = self.projector(vad_scores)
        logit = self.classifier(embedding)
        return embedding, logit

# Load labels
labels_raw = df['label'].apply(
    lambda x: 1.0 if x in [1, 'fake', 'Fake'] else 0.0
).values
labels_tensor = torch.tensor(labels_raw, dtype=torch.float32).to(device)

# Align VAD to 11844 rows (same as training pipeline)
n_img = 11844
text_vad_tensor = torch.tensor(
    text_vad[:n_img], dtype=torch.float32
).to(device)
labels_tensor = labels_tensor[:n_img]

# Dataset and loader
vad_dataset = TensorDataset(text_vad_tensor, labels_tensor)
vad_loader = DataLoader(vad_dataset, batch_size=64, shuffle=True)

# Train
vad_projector = VADProjector(input_dim=3, hidden_dim=16, output_dim=64).to(device)
optimizer = optim.Adam(vad_projector.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("\nTraining VAD projector...")
for epoch in range(20):
    vad_projector.train()
    epoch_loss = 0
    for vad_batch, label_batch in vad_loader:
        optimizer.zero_grad()
        _, logit = vad_projector(vad_batch)
        loss = criterion(logit.squeeze(), label_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20, Loss: {epoch_loss/len(vad_loader):.4f}")

# Generate trained embeddings
vad_projector.eval()
with torch.no_grad():
    full_vad_tensor = torch.tensor(
        text_vad[:n_img], dtype=torch.float32
    ).to(device)
    text_vad_embedding, _ = vad_projector(full_vad_tensor)

print(f"\nVAD embedding shape: {text_vad_embedding.shape}")

# Save
df_aligned = df.iloc[:n_img].copy()
df_aligned['text_vad_embedding'] = list(
    text_vad_embedding.cpu().numpy()
)

torch.save(
    text_vad_embedding.cpu(), 
    "Dataset/twitter/text_vad_embedding.pt"
)
print("Trained VAD embeddings saved as .pt tensor")

# Also save raw 3D VAD for reference
torch.save(
    torch.tensor(text_vad[:n_img], dtype=torch.float32),
    "Dataset/twitter/text_vad_raw.pt"
)
print(" Raw 3D VAD scores saved as text_vad_raw.pt")

df_aligned.to_pickle("Dataset/twitter/df_with_text_vad_embedding.pkl")
print(" Full DataFrame with trained VAD embeddings saved successfully ")

