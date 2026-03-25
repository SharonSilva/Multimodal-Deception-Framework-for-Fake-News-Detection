"""
1. Emotional Congruence Analysis (VAD-based)
2. Mismatch Vector Generation (preserved explicitly)
3. Fine-Grained Emotion Processing (VAD + Temporal)
4. Full data extraction pipeline
5. Integration with Adaptive Fusion Layer
6. End-to-end training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EMOTION-TO-VAD PROJECTION MODULE

class EmotionToVADProjector(nn.Module):
    """
    Projects discrete emotion probabilities to continuous VAD space.
    
    Formula from paper:
        e_vad = W_vad · e + b_vad, e_vad ∈ R³
    """
    def __init__(self, k_emotions, vad_dim=3, use_nonlinearity=True):
        super().__init__()
        self.k_emotions = k_emotions
        self.vad_dim = vad_dim
        self.use_nonlinearity = use_nonlinearity
        
        self.projection = nn.Linear(k_emotions, vad_dim)
        self._initialize_with_priors()
        
    def _initialize_with_priors(self):
        """Initialize with known emotion-VAD mappings from psychology literature"""
        if self.k_emotions == 7:
            # [angry, disgust, fear, happy, sad, surprise, neutral]
            emotion_vad_map = torch.tensor([
                [-0.6, 0.6, 0.5],   # angry
                [-0.6, 0.4, 0.3],   # disgust
                [-0.7, 0.6, -0.3],  # fear
                [0.8, 0.5, 0.5],    # happy
                [-0.7, -0.2, -0.3], # sad
                [0.4, 0.7, 0.0],    # surprise
                [0.0, 0.0, 0.0]     # neutral
            ], dtype=torch.float32)
            
            with torch.no_grad():
                self.projection.weight.data = emotion_vad_map.T
                self.projection.bias.data.zero_()
        
    def forward(self, emotion_probs):
        vad = self.projection(emotion_probs)
        if self.use_nonlinearity:
            vad = torch.tanh(vad)
        return vad


# COMPONENT 1: EMOTIONAL CONGRUENCE ANALYSIS

class EmotionalCongruenceScorer(nn.Module):
    """
    Computes alignment between text and image emotions using cosine similarity.
    High similarity → aligned emotions (credible)
    Low/negative similarity → conflicting emotions (deception signal)
    """
    def __init__(self, vad_dim=3):
        super().__init__()
        self.vad_dim = vad_dim
        
    def forward(self, vad_text, vad_image):
        congruence = F.cosine_similarity(vad_text, vad_image, dim=-1, eps=1e-8)
        return congruence.unsqueeze(-1)


# COMPONENT 2: MISMATCH VECTOR GENERATION

class MismatchVectorGenerator(nn.Module):
    """
    Generates mismatch vector that captures emotional contradictions.
    This vector is EXPLICITLY PRESERVED in the final representation.
    
    Formula: m = |vad_text - vad_image|
             m' = [m; affective_meta]
             v_mismatch = encoder(m')
    """
    def __init__(self, vad_dim=3, meta_affective_dim=128, mismatch_dim=128):
        super().__init__()
        # NO LayerNorm 
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(vad_dim + meta_affective_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, mismatch_dim)
        )
        
    def forward(self, vad_text, vad_image, affective_meta):
        m = torch.abs(vad_text - vad_image)
        m_prime = torch.cat([m, affective_meta], dim=-1)
        v_mismatch = self.mismatch_encoder(m_prime)
        return v_mismatch

# COMPONENT 3: FINE-GRAINED EMOTION PROCESSING

class FineGrainedEmotionProcessor(nn.Module):
    """
    Handles complex emotional states and temporal dynamics.
    Features: Temporal tracking, Mixed affect detection
    """
    def __init__(self, vad_dim=3, hidden_dim=64):
        super().__init__()
        self.temporal_encoder = nn.GRU(
            input_size=vad_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.mixed_affect_detector = nn.Sequential(
            nn.Linear(vad_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vad_text, vad_image, vad_sequence=None):
        # Temporal encoding
        if vad_sequence is not None and vad_sequence.size(1) > 1:
            output, h_n = self.temporal_encoder(vad_sequence)
            h_combined = torch.cat([h_n[0], h_n[1]], dim=-1)
            temporal_embedding = self.temporal_proj(h_combined)
        else:
            temporal_embedding = torch.zeros(
                vad_text.size(0), 64, device=vad_text.device
            )
        
        # Mixed affect detection
        vad_concat = torch.cat([vad_text, vad_image], dim=-1)
        mixed_affect_score = self.mixed_affect_detector(vad_concat)
        
        return temporal_embedding, mixed_affect_score



# COMPLETE EMOTION-GATED MECHANISM

class EmotionGatedMechanism(nn.Module):
    """
    Complete emotion-gated mechanism with dual input support:
    Pathway 1: Direct VAD inputs
    Pathway 2: Discrete emotions → VAD projection
    """
    def __init__(
        self, 
        vad_dim=3,
        meta_affective_dim=128,
        mismatch_dim=128,
        temporal_hidden=64,
        k_emotions_text=None,
        k_emotions_image=None
    ):
        super().__init__()
        
        self.use_emotion_projection_text = k_emotions_text is not None
        self.use_emotion_projection_image = k_emotions_image is not None
        
        if self.use_emotion_projection_text:
            self.emotion_to_vad_text = EmotionToVADProjector(k_emotions_text, vad_dim)
        
        if self.use_emotion_projection_image:
            self.emotion_to_vad_image = EmotionToVADProjector(k_emotions_image, vad_dim)
        
        self.congruence_scorer = EmotionalCongruenceScorer(vad_dim)
        self.mismatch_generator = MismatchVectorGenerator(
            vad_dim, meta_affective_dim, mismatch_dim
        )
        self.emotion_processor = FineGrainedEmotionProcessor(vad_dim, temporal_hidden)
        
        gate_input_dim = 1 + mismatch_dim + temporal_hidden + 1
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
        self.gamma = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self, 
        vad_text=None, 
        vad_image=None, 
        affective_meta=None,
        vad_sequence=None,
        emotion_probs_text=None,
        emotion_probs_image=None
    ):
        # Pathway selection
        if emotion_probs_text is not None and self.use_emotion_projection_text:
            vad_text_final = self.emotion_to_vad_text(emotion_probs_text)
        elif vad_text is not None:
            vad_text_final = vad_text
        else:
            raise ValueError("Must provide either vad_text or emotion_probs_text")
        
        if emotion_probs_image is not None and self.use_emotion_projection_image:
            vad_image_final = self.emotion_to_vad_image(emotion_probs_image)
        elif vad_image is not None:
            vad_image_final = vad_image
        else:
            raise ValueError("Must provide either vad_image or emotion_probs_image")
        
        # Core processing
        congruence = self.congruence_scorer(vad_text_final, vad_image_final)
        v_mismatch = self.mismatch_generator(vad_text_final, vad_image_final, affective_meta)
        temporal_features, mixed_affect_score = self.emotion_processor(
            vad_text_final, vad_image_final, vad_sequence
        )
        
        gate_input = torch.cat([
            congruence, v_mismatch, temporal_features, mixed_affect_score
        ], dim=-1)
        emotion_gate = self.gating_network(gate_input)
        
        return (
            emotion_gate, v_mismatch, congruence, 
            temporal_features, mixed_affect_score,
            vad_text_final, vad_image_final
        )


# EMOTION-AWARE FUSION LAYER


class EmotionAwareFusionLayer(nn.Module):
    """
    Integrates Emotion-Gated Mechanism with Adaptive Fusion.
    Key innovation: Explicitly preserves mismatch vector
    Formula: z_out = [z_fused; γ*v_mismatch]
    """
    def __init__(
        self,
        d_text=128,
        d_image=1024,
        d_meta=128,
        d_common=256,
        vad_dim=3,
        meta_affective_dim=128,
        mismatch_dim=128,
        k_emotions_text=None,
        k_emotions_image=None
    ):
        super().__init__()
        
        self.proj_text = nn.Linear(d_text, d_common)
        self.proj_image = nn.Linear(d_image, d_common)
        self.proj_meta = nn.Linear(d_meta, d_common)
        
        self.emotion_gate = EmotionGatedMechanism(
            vad_dim=vad_dim,
            meta_affective_dim=meta_affective_dim,
            mismatch_dim=mismatch_dim,
            temporal_hidden=64,
            k_emotions_text=k_emotions_text,
            k_emotions_image=k_emotions_image
        )
        
        self.norm_fused = nn.LayerNorm(d_common)
        self.norm_mismatch = nn.LayerNorm(mismatch_dim)
        
    def forward(
        self, 
        h_text, h_image, h_meta,
        affective_meta,
        vad_text=None,
        vad_image=None,
        vad_sequence=None,
        emotion_probs_text=None,
        emotion_probs_image=None
    ):
        # Project to common space
        z_text = self.proj_text(h_text)
        z_image = self.proj_image(h_image)
        z_meta = self.proj_meta(h_meta)
        
        # Emotion-gated mechanism
        outputs = self.emotion_gate(
            vad_text=vad_text,
            vad_image=vad_image,
            affective_meta=affective_meta,
            vad_sequence=vad_sequence,
            emotion_probs_text=emotion_probs_text,
            emotion_probs_image=emotion_probs_image
        )
        
        emotion_weights, v_mismatch, congruence, temporal_feats, mixed_affect, \
            vad_text_final, vad_image_final = outputs
        
        # Apply gating
        z_text_gated = z_text * emotion_weights[:, 0:1]
        z_image_gated = z_image * emotion_weights[:, 1:2]
        z_meta_gated = z_meta * emotion_weights[:, 2:3]
        
        # Fuse modalities
        z_fused = z_text_gated + z_image_gated + z_meta_gated
        
        # Normalize parts
        z_fused = self.norm_fused(z_fused)
        v_mismatch_norm = self.norm_mismatch(self.emotion_gate.gamma * v_mismatch)

        
        m_prime = torch.cat([temporal_feats, mixed_affect], dim=-1)

        z_aug = torch.cat([
            z_fused,          
            v_mismatch_norm,  
            m_prime           
        ], dim=-1)
        
        intermediates = {
            'z_fused': z_fused,
            'v_mismatch': v_mismatch,
            'v_mismatch_norm': v_mismatch_norm,
            'm_prime': m_prime,
            'emotion_weights': emotion_weights,
            'congruence': congruence,
            'temporal_features': temporal_feats,
            'mixed_affect_score': mixed_affect,
            'z_text_gated': z_text_gated,
            'z_image_gated': z_image_gated,
            'z_meta_gated': z_meta_gated,
            'vad_text': vad_text_final,
            'vad_image': vad_image_final
        }
        intermediates['gamma'] = self.emotion_gate.gamma
        intermediates['z_aug'] = z_aug

        return z_aug, intermediates
    
class EmotionAwareFakeNewsDetector(nn.Module):
        def __init__(
            self,
            d_text=128,
            d_image=1024,
            d_meta=128,
            d_common=256,
            vad_dim=3,
            meta_affective_dim=128,
            mismatch_dim=128,
            temporal_hidden=64,
            num_classes=1
        ):
            super().__init__()

            # Fusion output size
            self.fusion_output_dim = d_common + mismatch_dim + (temporal_hidden + 1)

            #  Use your EXISTING fusion module here
            self.fusion_layer = EmotionAwareFusionLayer(
                d_text=d_text,
                d_image=d_image,
                d_meta=d_meta,
                d_common=d_common,
                vad_dim=vad_dim,
                meta_affective_dim=meta_affective_dim,
                mismatch_dim=mismatch_dim,
            )

            #  Classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_output_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )

        def forward(
            self, 
            h_text, h_image, h_meta,
            affective_meta,
            vad_text=None,
            vad_image=None,
            vad_sequence=None,
            emotion_probs_text=None,
            emotion_probs_image=None
        ):
            #  Fusion
            z_aug, intermediates = self.fusion_layer(
                h_text, h_image, h_meta,
                affective_meta,
                vad_text=vad_text,
                vad_image=vad_image,
                vad_sequence=vad_sequence,
                emotion_probs_text=emotion_probs_text,
                emotion_probs_image=emotion_probs_image
            )

            #  Classification
            logits = self.classifier(z_aug)

            return logits, intermediates



# DATA EXTRACTION PIPELINE


class VADDataExtractor:
    """
    Extracts 3D VAD from trained models and prepares data for emotion-gated mechanism.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_face_vad(self, image_paths, model_path):
        """Extract 3D VAD from trained AffectNet ViTForVAD model"""
        from train_vit_vad import ViTForVAD

        print("Loading AffectNet model...")
        model = ViTForVAD().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        all_vad = []
        print("Extracting face VAD...")

        with torch.no_grad():
            for img_path in tqdm(image_paths):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    vad_pred, _ = model(img_tensor)
                    all_vad.append(vad_pred.cpu().numpy()[0])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    all_vad.append(np.array([0.0, 0.0, 0.0]))

        return np.array(all_vad)

    def extract_scene_vad(self, scene_csv_path):
        """Extract scene VAD from CSV"""
        print("Loading scene VAD...")
        df_scene = pd.read_csv(scene_csv_path)
        
        if all(col in df_scene.columns for col in ['valence', 'arousal', 'dominance']):
            vad_scene = df_scene[['valence', 'arousal', 'dominance']].values
        else:
            print("Warning: Scene VAD columns not found, using neutral values")
            vad_scene = np.full((len(df_scene), 3), 0.5)
        
        return vad_scene
    
    def prepare_complete_vad_data(
        self,
        df,
        affectnet_model_path=None,
        scene_csv_path=None,
        use_face=True,
        use_scene=True,
        image_folder=None
    ):
        """
        Prepare complete VAD data from all sources, handling missing images
        and ensuring consistent array shapes.
        """
        # Text VAD  
        print("Loading text VAD...")
        vad_text = torch.tensor(
            df[['text_valence', 'text_arousal', 'text_dominance']].values,
            dtype=torch.float32
        )

        vad_components = []

        # Face VAD
        if use_face and affectnet_model_path and image_folder:
            from train_vit_vad import ViTForVAD
            print("Loading face emotion model...")
            model = ViTForVAD().to(self.device)
            state_dict = torch.load(affectnet_model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            image_paths = [
                os.path.join(image_folder, f"{img_id}.jpg") for img_id in df['image_id']
            ]
            vad_face = []
            
            print("Extracting face VAD...")
            with torch.no_grad():
                for img_path in tqdm(image_paths, desc="Processing faces"):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                        vad_pred, _ = model(img_tensor)
                        vad_face.append(vad_pred.cpu().numpy()[0])
                    except Exception as e:
                        # Silent fallback for missing images 
                        vad_face.append(np.array([0.0, 0.0, 0.0]))

            vad_face = np.array(vad_face)
            print(f" Extracted face VAD: {vad_face.shape}")
            vad_components.append(vad_face)

        # Scene VAD 
        if use_scene and scene_csv_path:
            vad_scene = self.extract_scene_vad(scene_csv_path)
            
            # Handle size mismatch
            if len(vad_scene) != len(df):
                print(f" Scene VAD length ({len(vad_scene)}) != posts ({len(df)})")
                
                if len(vad_scene) < len(df):
                    # Pad with zeros
                    pad_len = len(df) - len(vad_scene)
                    padding = np.zeros((pad_len, 3))
                    vad_scene = np.vstack([vad_scene, padding])
                    print(f"   Padded scene VAD with {pad_len} neutral values")
                else:
                    # Trim
                    vad_scene = vad_scene[:len(df)]
                    print(f"   Trimmed scene VAD to {len(df)}")
            
            print(f" Scene VAD ready: {vad_scene.shape}")
            vad_components.append(vad_scene)

        #  Combine face and scene VAD 
        if len(vad_components) > 0:
            # Ensure all arrays are same length
            target_len = len(df)
            vad_components_aligned = []
            
            for i, vad_comp in enumerate(vad_components):
                if len(vad_comp) != target_len:
                    print(f" Component {i} length mismatch, fixing...")
                    if len(vad_comp) < target_len:
                        pad_len = target_len - len(vad_comp)
                        vad_comp = np.vstack([vad_comp, np.zeros((pad_len, 3))])
                    else:
                        vad_comp = vad_comp[:target_len]
                vad_components_aligned.append(vad_comp)
            
            # Average face and scene VAD
            vad_image = np.mean(vad_components_aligned, axis=0)
            print(f" Combined image VAD (face + scene): {vad_image.shape}")
        else:
            print(" No image VAD sources, using neutral values")
            vad_image = np.full((len(df), 3), 0.5)

        vad_image = torch.tensor(vad_image, dtype=torch.float32)

        # 5. Affective metadata (FIXED) 
        print("Loading affective metadata...")
        try:
            affective_meta = np.load("Dataset/affectnet/affective_embedding.npy")
            
            # Handle size mismatch
            if len(affective_meta) < len(df):
                print(f" Affective embeddings ({len(affective_meta)}) < posts ({len(df)})")
                print("   Padding with mean values...")
                mean_embedding = affective_meta.mean(axis=0)
                pad_size = len(df) - len(affective_meta)
                padding = np.tile(mean_embedding, (pad_size, 1))
                affective_meta = np.vstack([affective_meta, padding])
            elif len(affective_meta) > len(df):
                print(f" Affective embeddings ({len(affective_meta)}) > posts ({len(df)})")
                print("   Trimming to match...")
                affective_meta = affective_meta[:len(df)]
            
            affective_meta = torch.tensor(affective_meta, dtype=torch.float32)
            print(f" Affective metadata: {affective_meta.shape}")
            
        except Exception as e:
            print(f" Affective embeddings error: {e}")
            print("   Using random values as fallback...")
            affective_meta = torch.randn(len(df), 128)

        # 6. Optional: Create temporal sequences per user 
        if 'username' in df.columns:
            print("Creating user-level VAD sequences...")
            user_sequences = {}
            for user in df['username'].unique():
                user_mask = df['username'] == user
                user_vad = vad_text[user_mask]
                if len(user_vad) > 1:
                    user_sequences[user] = user_vad
            print(f" Created sequences for {len(user_sequences)} users")
        else:
            user_sequences = None

        #  7. Prepare output dictionary 
        output_dict = {
            'vad_text': vad_text,
            'vad_image': vad_image,
            'affective_meta': affective_meta,
            'vad_sequence': vad_text.unsqueeze(1), 
            'user_sequences': user_sequences
        }
        
        #  Save prepared data for future use
        save_path = 'Dataset/twitter/prepared_vad_data.pt'
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(output_dict, save_path)
            print(f" Saved prepared VAD data to {save_path}")
        except Exception as e:
            print(f" Could not save VAD data: {e}")
        
        return output_dict


# COMPLETE TRAINING PIPELINE


def train_emotion_aware_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=10,
    device='cpu'
):
    """
    Train emotion-aware fusion model with mismatch-aware losses.
    """
    criterion_main = nn.BCEWithLogitsLoss()
    criterion_mismatch = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            h_text = batch['text_features'].to(device)
            h_image = batch['image_features'].to(device)
            h_meta = batch['metadata_features'].to(device)
            vad_text = batch['vad_text'].to(device)
            vad_image = batch['vad_image'].to(device)
            affective_meta = batch['affective_meta'].to(device)
            labels = batch['labels'].to(device).float()
            
            optimizer.zero_grad()
            
            
            # Main classification loss
            logits, intermediates = model(
                h_text, h_image, h_meta,
                affective_meta,
                vad_text = vad_text,
                vad_image = vad_image
            )
            loss_main = criterion_main(logits.squeeze(), labels)
            
            # Mismatch magnitude loss (encourage large mismatch for fake news)
            v_mismatch = intermediates['v_mismatch']
            congruence = intermediates['congruence']
            mismatch_magnitude = torch.norm(v_mismatch, dim=1)
            target_magnitude = labels * 2.0  # High for fake (1), low for real (0)
            loss_mismatch = criterion_mismatch(mismatch_magnitude, target_magnitude)
            
            # Congruence loss (low congruence should correlate with fake news)
            congruence = intermediates['congruence'].squeeze()
            loss_congruence = criterion_mismatch(1.0 - congruence, labels)
            
            # Combined loss
            loss = loss_main + 0.1 * loss_mismatch + 0.05 * loss_congruence
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                h_text = batch['text_features'].to(device)
                h_image = batch['image_features'].to(device)
                h_meta = batch['metadata_features'].to(device)
                vad_text = batch['vad_text'].to(device)
                vad_image = batch['vad_image'].to(device)
                affective_meta = batch['affective_meta'].to(device)
                labels = batch['labels'].to(device).float()

                logits, intermediates  = model(
                h_text, h_image, h_meta,
                affective_meta,
                vad_text = vad_text,
                vad_image = vad_image
                )
                loss = criterion_main(logits.squeeze(), labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_emotion_aware_model.pth")
            print("   Best model saved!")
    
    return model


# USAGE EXAMPLE

if __name__ == "__main__":
    print("="*70)
    print("COMPLETE EMOTION-GATED MECHANISM - PRODUCTION VERSION")
    print("="*70)
    
    # STEP 1: Extract VAD from all sources 
    print("\n Step 1: Extracting VAD data...")
    
    df = pd.read_pickle("Dataset/twitter/df_with_text_emotions_vad.pkl")
    
    extractor = VADDataExtractor(device=device)
    vad_data = extractor.prepare_complete_vad_data(
        df=df,
        affectnet_model_path="vit_affectnet_vad.pth",
        scene_csv_path="Dataset/twitter/scene_emotions_vad_proj.csv",
        use_face=True,
        use_scene=True,
        image_folder="Dataset/twitter/images_train"
    )
    
    print(f" VAD data prepared:")
    print(f"   Text VAD: {vad_data['vad_text'].shape}")
    print(f"   Image VAD: {vad_data['vad_image'].shape}")
    print(f"   Affective meta: {vad_data['affective_meta'].shape}")
    
    #  Initialize model 
    print("\n Step 2: Initializing model...")
    
    model = EmotionAwareFakeNewsDetector(
        d_text=128,
        d_image=1024,
        d_meta=128,
        d_common=256,
        vad_dim=3,
        meta_affective_dim=128,
        mismatch_dim=128
    ).to(device)
    
    print(f" Model initialized")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    #  Test forward pass
    print("\n Step 3: Testing forward pass...")
    
    batch_size = 8
    h_text = torch.randn(batch_size, 128).to(device)
    h_image = torch.randn(batch_size, 1024).to(device)
    h_meta = torch.randn(batch_size, 128).to(device)
    
    logits, intermediates = model(
        h_text, h_image, h_meta,
        affective_meta=vad_data['affective_meta'][:batch_size].to(device),
        vad_text=vad_data['vad_text'][:batch_size].to(device),
        vad_image=vad_data['vad_image'][:batch_size].to(device)
    )
    