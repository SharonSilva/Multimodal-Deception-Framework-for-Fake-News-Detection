import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚡ Using device: {DEVICE}")


LABELS_CSV = "Dataset/affectnet/labels.csv"  
TRAIN_DIR = "Dataset/affectnet/Train"
SAVE_MODEL_PATH = "vit_affectnet_vad.pth"
VAD_CACHE = "Dataset/affectnet/labels_with_vad.csv"


emotion_to_vad = {
    0: [0.5, 0.3, 0.5],  # neutral
    1: [0.8, 0.6, 0.7],  # happy
    2: [0.2, 0.6, 0.3],  # sad
    3: [0.1, 0.7, 0.2],  # angry
    4: [0.9, 0.8, 0.7],  # surprise
    5: [0.3, 0.8, 0.4],  # fear
    6: [0.3, 0.5, 0.3],  # disgust
    7: [0.5, 0.4, 0.5],  # contempt
}

def compute_vad_from_emotion_label(label):
    vad = np.array(emotion_to_vad.get(label, [0.5, 0.5, 0.5]), dtype=np.float32)
    vad = (vad - 0.5) * 2  # normalize [-1,1]
    vad += np.random.normal(0, 0.02, size=3)
    return np.clip(vad, -1, 1)


labels_df = pd.read_csv(LABELS_CSV)
labels_df['exists'] = labels_df['pth'].apply(lambda x: os.path.exists(os.path.join(TRAIN_DIR, os.path.basename(x))))
labels_df = labels_df[labels_df['exists']].reset_index(drop=True)


if not os.path.exists(VAD_CACHE):
    print(" Generating VAD values...")
    vad_values = [compute_vad_from_emotion_label(row['label']) for _, row in tqdm(labels_df.iterrows(), total=len(labels_df))]
    labels_df[['valence', 'arousal', 'dominance']] = pd.DataFrame(vad_values)
    labels_df.to_csv(VAD_CACHE, index=False)
else:
    labels_df = pd.read_csv(VAD_CACHE)
    print(f" Loaded precomputed VADs from {VAD_CACHE}")


train_df, val_df = train_test_split(labels_df, test_size=0.1, random_state=42)
train_df.to_csv("Dataset/affectnet/train_vad.csv", index=False)
val_df.to_csv("Dataset/affectnet/val_vad.csv", index=False)


transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


class AffectNetDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(row['pth']))
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB",(224,224),(0,0,0))
        if self.transform:
            img = self.transform(img)
        vad = compute_vad_from_emotion_label(row['label'])
        return img, torch.tensor(vad, dtype=torch.float32)


class ViTForVAD(nn.Module):
    def __init__(self, vad_dim=3):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.vad_head = nn.Linear(768, vad_dim)

    def forward(self, x):
        feats = self.vit(x)
        vad_pred = self.vad_head(feats)
        return vad_pred, feats


class VADProjector(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self, vad_scores):
        return self.projector(vad_scores)

    def train_supervised(self, vad_tensor, label_tensor, epochs=20, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        classifier = nn.Linear(64, 1).to(vad_tensor.device)
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(classifier.parameters()), lr=lr
        )
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            emb = self.forward(vad_tensor)
            loss = criterion(classifier(emb).squeeze(), label_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"  VAD projector epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")
        self.eval()


def train_model(train_loader, val_loader, model, epochs=3, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds, _ = model(imgs)  # preds = 3D VAD
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=train_loss / (loop.n + 1))

        model.eval()
        val_loss = 0
        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for imgs, labels in loop:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds, _ = model(imgs)
                val_loss += loss_fn(preds, labels).item()
                loop.set_postfix(val_loss=val_loss / (loop.n + 1))
        print(f" Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    return model

if __name__ == "__main__":
    train_dataset = AffectNetDataset("Dataset/affectnet/train_vad.csv", TRAIN_DIR, transform=transform_train)
    val_dataset = AffectNetDataset("Dataset/affectnet/val_vad.csv", TRAIN_DIR, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize models
    model = ViTForVAD().to(DEVICE)
    vad_projector = VADProjector(input_dim=3, hidden_dim=16, output_dim=64).to(DEVICE)

    # Train ViT → 3D VAD
    model = train_model(train_loader, val_loader, model, epochs=3, lr=1e-4)
    
    all_vad_preds = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(DataLoader(
            AffectNetDataset(VAD_CACHE, TRAIN_DIR, transform=transform_val),
            batch_size=32, shuffle=False, num_workers=0
        ), desc="Collecting VAD predictions"):
            imgs = imgs.to(DEVICE)
            vad_pred, _ = model(imgs)
            all_vad_preds.extend(vad_pred.cpu())
            
    assert len(all_vad_preds) == len(labels_df), \
        f"Mismatch: {len(all_vad_preds)} predictions vs {len(labels_df)} labels"
    # Save models
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    # Train VAD projector with supervision signal
    print("Training VAD projector...")
    labels_tensor = torch.tensor(
        labels_df['label'].apply(
            lambda x: 1.0 if x in [1, 3, 4, 5] else 0.0
            # happy, angry, surprise, fear = high arousal = 1
    ).values[:len(all_vad_preds)],
        dtype=torch.float32
    ).to(DEVICE)

    vad_preds_tensor = torch.stack(all_vad_preds).to(DEVICE)
    vad_projector.train_supervised(vad_preds_tensor, labels_tensor)

    torch.save(vad_projector.state_dict(), "vad_projector.pth")
    print(" VAD model and projector saved.")

    # Extract 64D embeddings for downstream tasks
    dataset_loader = DataLoader(AffectNetDataset(VAD_CACHE, TRAIN_DIR, transform=transform_val), batch_size=16, shuffle=False, num_workers=0)
    all_embeddings = []
    model.eval()
    vad_projector.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(dataset_loader, desc="Extracting VAD embeddings"):
            imgs = imgs.to(DEVICE)
            vad_pred, _ = model(imgs)           # 3D VAD
            embeddings = vad_projector(vad_pred)  # 64D projected embeddings
            all_embeddings.extend(embeddings.cpu().numpy())
    labels_df['image_vad_embedding'] = all_embeddings
    labels_df.to_pickle("Dataset/affectnet/df_with_image_vad_embedding.pkl")
    print(" Image VAD embeddings saved.")
