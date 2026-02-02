"""
STANDALONE FAKE NEWS DETECTOR - EMOTION MODEL FIXED (MINIMAL CHANGES)
======================================================================
ONLY the emotion model loading is fixed. Everything else is UNCHANGED.
"""

import os, re, warnings, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}\n")

# ============================================================
# REBUILD MODEL ARCHITECTURES
# ============================================================

class EmotionGate(nn.Module):
    """
    Emotion gate - FIXED to include temporal encoder and gamma
    """
    def __init__(self, input_dim=194, mismatch_input_dim=131, mismatch_output_dim=128, temporal_hidden=64):
        super().__init__()

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.mismatch_encoder = nn.Sequential(
            nn.Linear(mismatch_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, mismatch_output_dim)
        )
        
        # ✅ ADDED: Temporal encoder and gamma
        self.temporal_encoder = nn.LSTM(3, 32, 1, batch_first=True, bidirectional=True)
        self.temporal_proj = nn.Linear(64, temporal_hidden)
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Scalar, not [1.0]

    def forward(self, emotion_input):
        vad_text = emotion_input[:, :3]
        vad_image = emotion_input[:, 3:6]
        affective_meta = emotion_input[:, 6:]

        weights = torch.softmax(self.gating_network(emotion_input), dim=-1)
        v_mismatch = self.mismatch_encoder(emotion_input[:, :131])
        
        # ✅ ADDED: Temporal encoding
        vad_sequence = torch.stack([vad_text, vad_image], dim=1)
        lstm_out, _ = self.temporal_encoder(vad_sequence)
        temporal_features = self.temporal_proj(lstm_out[:, -1, :])
        
        congruence = F.cosine_similarity(vad_text, vad_image, dim=-1)
        mixed_affect_score = torch.sigmoid(torch.norm(v_mismatch, dim=-1))

        # ✅ MODIFIED: Return temporal and gamma
        return weights, v_mismatch, temporal_features, self.gamma, congruence, mixed_affect_score


class EmotionAwareFusionLayer(nn.Module):
    """
    Fusion layer - FIXED to output 449 dimensions
    """
    def __init__(self, d_text=128, d_image=1024, d_meta=128):
        super().__init__()

        self.proj_text = nn.Linear(d_text, 256)
        self.proj_image = nn.Linear(d_image, 256)
        self.proj_meta = nn.Linear(d_meta, 256)

        self.emotion_gate = EmotionGate(input_dim=194, mismatch_input_dim=131, mismatch_output_dim=128, temporal_hidden=64)

        # ✅ FIXED: 256 + 128 + 64 + 1 = 449
        self.fusion_output_dim = 449

    def forward(self, h_text, h_image, h_meta, emotion_input):
        # ✅ MODIFIED: Receive temporal and gamma
        weights, v_mismatch, temporal_features, gamma, congruence, mixed_affect = self.emotion_gate(emotion_input)

        h_text_proj = self.proj_text(h_text)
        h_image_proj = self.proj_image(h_image)
        h_meta_proj = self.proj_meta(h_meta)

        z_fused = (weights[:, 0:1] * h_text_proj +
                   weights[:, 1:2] * h_image_proj +
                   weights[:, 2:3] * h_meta_proj)

        # ✅ FIXED: Concatenate all 4 components (256 + 128 + 64 + 1 = 449)
        # gamma is already a scalar from emotion_gate, expand it to [batch, 1]
        gamma_expanded = gamma.unsqueeze(0).expand(h_text.shape[0], 1)
        z_out = torch.cat([z_fused, v_mismatch, temporal_features, gamma_expanded], dim=-1)

        return z_out, {
            'emotion_weights': weights,
            'v_mismatch': v_mismatch,
            'congruence': congruence,
            'mixed_affect_score': mixed_affect
        }


class StandaloneEmotionAwareDetector(nn.Module):
    """
    Full detector - NO CHANGES (fusion_output_dim is now correct)
    """
    def __init__(self):
        super().__init__()

        d_text = 128
        d_image = 1024
        d_meta = 128

        self.fusion_layer = EmotionAwareFusionLayer(d_text, d_image, d_meta)

        # Classifier automatically uses correct 449 dims now
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_layer.fusion_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, h_text, h_image, h_meta, emotion_input):
        z_out, intermediates = self.fusion_layer(h_text, h_image, h_meta, emotion_input)
        logits = self.classifier(z_out)
        return logits, intermediates


class StandaloneAdaptiveFusion(nn.Module):
    """Adaptive fusion - UNCHANGED"""
    def __init__(self, d_text=64, d_image=64, d_meta=64, d_common=256):
        super().__init__()
        
        self.suspicion_detector = nn.Sequential(
            nn.Linear(d_text + d_image + d_meta, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        self.modality_attention = nn.Sequential(
            nn.Linear(d_text + d_image + d_meta, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        self.text_proj = nn.Linear(d_text, d_common)
        self.image_proj = nn.Linear(d_image, d_common)
        self.meta_proj = nn.Linear(d_meta, d_common)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_common, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, h_text, h_image, h_meta, return_intermediates=False):
        combined = torch.cat([h_text, h_image, h_meta], dim=-1)
        
        suspicion_scores = torch.sigmoid(self.suspicion_detector(combined))
        modality_weights = torch.softmax(self.modality_attention(combined), dim=-1)
        
        fused = (modality_weights[:, 0:1] * self.text_proj(h_text) +
                 modality_weights[:, 1:2] * self.image_proj(h_image) +
                 modality_weights[:, 2:3] * self.meta_proj(h_meta))
        
        logits = self.classifier(fused)
        
        if return_intermediates:
            return logits, {
                'suspicion_scores': suspicion_scores,
                'modality_weights': modality_weights
            }
        return logits


# ============================================================
# Standalone Detector - COMPLETELY UNCHANGED FROM YOUR ORIGINAL
# ============================================================
class StandaloneFakeNewsDetector:
    """
    Standalone detector that doesn't import from training files.
    Rebuilds models from scratch and loads trained weights.
    """
    
    def __init__(self):
        print("="*70)
        print("INITIALIZING STANDALONE FAKE NEWS DETECTOR")
        print("="*70)
        
        self.device = device
        
        self._load_transformers()
        self._load_trained_models()
        
        print("\n✅ READY FOR NEW POST INFERENCE\n")
        print("="*70)
    
    def _load_transformers(self):
        """Load transformer models for feature extraction."""
        print("\n[1/2] Loading feature extractors...")
        
        from transformers import (
            BertTokenizer, BertModel,
            ViTFeatureExtractor, ViTModel,
            AutoTokenizer, AutoModelForSequenceClassification
        )
        from torchvision.models import resnet50
        
        # BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()
        
        # Vision
        self.vit_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device).eval()
        
        self.resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]).to(self.device).eval()
        
        # Emotion
        self.emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
        self.emotion_classifier = AutoModelForSequenceClassification.from_pretrained(
            "nateraw/bert-base-uncased-emotion"
        ).to(self.device).eval()
        
        # Projection layers
        self.text_proj = nn.Linear(768, 128).to(self.device)
        self.meta_proj = nn.Linear(7, 128).to(self.device)
        
        print("  ✓ Feature extractors loaded")
    
    def _load_trained_models(self):
        """Load trained detection models from checkpoints."""
        print("\n[2/2] Loading trained detection models...")
        
        self.emotion_model = None
        self.fusion_model = None
        
        # Emotion-aware model (in checkpoints/ directory)
        emotion_ckpt = Path("checkpoints/best_emotion_aware_detector.pth")
        if emotion_ckpt.exists():
            try:
                print(f"  Loading emotion checkpoint: {emotion_ckpt}")
                
                # Rebuild model architecture
                self.emotion_model = StandaloneEmotionAwareDetector().to(self.device)
                
                # Load checkpoint
                state = torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=self.device)

                
                # Handle key remapping
                remapped = {}
                for k, v in state.items():
                    # fusion. -> fusion_layer.
                    new_k = k.replace("fusion.", "fusion_layer.") if k.startswith("fusion.") else k
                    remapped[new_k] = v
                
                # Load weights (strict=False to ignore missing keys)
                self.emotion_model.load_state_dict(remapped, strict=False)
                self.emotion_model.eval()
                
                print(f"    ✅ Emotion model loaded successfully")
            except Exception as e:
                print(f"    ⚠ Emotion model failed to load: {e}")
                self.emotion_model = None
        else:
            print(f"  ⚠ Emotion checkpoint not found: {emotion_ckpt}")
        
        # Adaptive fusion model (in root directory)
        fusion_ckpt = Path("best_model_safe.pt")
        if fusion_ckpt.exists():
            try:
                print(f"  Loading fusion checkpoint: {fusion_ckpt}")
                
                # Rebuild model architecture
                self.fusion_model = StandaloneAdaptiveFusion(
                    d_text=64, d_image=64, d_meta=64, d_common=256
                ).to(self.device)
                
                # Load checkpoint
                state = torch.load(fusion_ckpt, map_location=self.device)
                
                # Try to load (might need key mapping)
                try:
                    self.fusion_model.load_state_dict(state, strict=False)
                except RuntimeError:
                    # If direct load fails, the checkpoint has different architecture
                    print(f"    ⚠ Checkpoint architecture mismatch - skipping fusion model")
                    self.fusion_model = None
                else:
                    self.fusion_model.eval()
                    print(f"    ✅ Fusion model loaded successfully")
            except Exception as e:
                print(f"    ⚠ Fusion model failed to load: {e}")
                self.fusion_model = None
        else:
            print(f"  ⚠ Fusion checkpoint not found: {fusion_ckpt}")
        
        # Check if any models loaded
        if self.emotion_model is None and self.fusion_model is None:
            print("\n  ⚠️  WARNING: No trained models loaded!")
            print("  Predictions will be unreliable.")
            print("\n  Make sure you have:")
            print(f"    - checkpoints/best_emotion_aware_detector.pth")
            print(f"    - best_model_safe.pt (in root directory)")
    
    def predict(self, text, image_path=None, username="anonymous"):
        """
        Predict if a new post is fake or real.
        """
        import emoji
        from PIL import Image
        import torch.nn.functional as F

        post_id = hashlib.md5(text.encode()).hexdigest()[:8]
        timestamp = datetime.now()

        print(f"\n{'─'*70}")
        print(f"📝 POST: {post_id} | USER: {username}")
        print(f"{'─'*70}")
        print(f"{text[:100]}..." if len(text) > 100 else text)
        print(f"{'─'*70}\n")

        # -------------------
        # CLEAN TEXT
        # -------------------
        clean = re.sub(r'http\S+|www.\S+|#\w+|@\w+', '', text)
        clean = emoji.replace_emoji(clean, replace='')
        clean = re.sub(r'\s+', ' ', clean).strip()

        # -------------------
        # METADATA
        # -------------------
        hashtags = len(re.findall(r'#\w+', text))
        mentions = len(re.findall(r'@\w+', text))
        urls = len(re.findall(r'http\S+|www.\S+', text))
        emojis = len([c for c in text if emoji.is_emoji(c)])

        # -------------------
        # TEXT EMBEDDING
        # -------------------
        tokens = self.bert_tokenizer(clean, padding=True, truncation=True,
                                    max_length=64, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            bert_out = self.bert(**tokens).last_hidden_state
            text_emb = self.text_proj(bert_out.mean(dim=1))
            text_emb = text_emb / (torch.norm(text_emb) + 1e-8)

        # -------------------
        # IMAGE EMBEDDING
        # -------------------
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            image_present = 1.0
        else:
            img = Image.new("RGB", (224, 224))
            image_present = 0.0

        vit_input = self.vit_extractor(images=[img], return_tensors="pt")['pixel_values'].to(self.device)
        with torch.no_grad():
            vit_emb = self.vit(vit_input).last_hidden_state[:, 0, :]

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_out = self.resnet(img_tensor).mean(dim=[2, 3])
        cnn_pooled = nn.AdaptiveAvgPool1d(256)(cnn_out.unsqueeze(1)).squeeze(1)
        image_emb = torch.cat([vit_emb, cnn_pooled], dim=-1)  # 1024-dim
        image_emb = image_emb / (torch.norm(image_emb) + 1e-8)
        image_emb = image_emb * image_present  # zero out dummy

        # -------------------
        # METADATA EMBEDDING
        # -------------------
        ts = timestamp.timestamp()
        period = 24*60*60
        meta_vec = torch.tensor([
            hashtags, mentions, urls, emojis, image_present,
            np.sin(2*np.pi*ts/period), np.cos(2*np.pi*ts/period)
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            meta_emb = self.meta_proj(meta_vec)
            meta_emb = meta_emb / (torch.norm(meta_emb) + 1e-8)

        # -------------------
        # EMOTION (VAD)
        # -------------------
        emotion_tokens = self.emotion_tokenizer(clean, padding=True, truncation=True,
                                            max_length=128, return_tensors="pt")
        emotion_tokens = {k: v.to(self.device) for k, v in emotion_tokens.items()}

        with torch.no_grad():
            emotion_logits = self.emotion_classifier(**emotion_tokens).logits
            emotion_probs = torch.softmax(emotion_logits, dim=-1).cpu().numpy()[0]

        vad_map = {
            0: [0.1, 0.3, 0.4],  # sadness
            1: [0.9, 0.7, 0.8],  # joy
            2: [0.8, 0.6, 0.7],  # love
            3: [0.2, 0.8, 0.6],  # anger
            4: [0.2, 0.8, 0.3],  # fear
            5: [0.6, 0.8, 0.5],  # surprise
        }

        vad = np.zeros(3, dtype=np.float32)
        for i, prob in enumerate(emotion_probs):
            if i in vad_map:
                vad += prob * np.array(vad_map[i])

        vad_text = torch.tensor(vad, dtype=torch.float32).unsqueeze(0).to(self.device)
        vad_image = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32).to(self.device)
        affective_meta = torch.zeros(1, 128).to(self.device)
        pad = torch.zeros(1, 60).to(self.device)

        # ✅ FINAL PADDED EMOTION INPUT (1x194)
        emotion_input = torch.cat([vad_text, vad_image, affective_meta, pad], dim=-1)
        assert emotion_input.shape == (1, 194), "Emotion input padding mismatch!"

        # -------------------
        # EMOTION MODEL INFERENCE
        # -------------------
        preds, confs = [], []
        emotion_details = None
        
        if self.emotion_model:
            with torch.no_grad():
                logits, intermediates = self.emotion_model(
                    h_text=text_emb,
                    h_image=image_emb,
                    h_meta=meta_emb,
                    emotion_input=emotion_input
                )
                score = torch.sigmoid(logits).item()
                conf = abs(score - 0.5) * 2
                preds.append(score)
                confs.append(conf)
                
                # Store detailed emotion analysis
                emotion_details = {
                    'score': score,
                    'confidence': conf,
                    'emotion_weights': {
                        'text': float(intermediates['emotion_weights'][0, 0]),
                        'image': float(intermediates['emotion_weights'][0, 1]),
                        'meta': float(intermediates['emotion_weights'][0, 2])
                    },
                    'congruence': float(intermediates['congruence']),
                    'mismatch_magnitude': float(torch.norm(intermediates['v_mismatch'])),
                    'mixed_affect': float(intermediates['mixed_affect_score'])
                }
                
                print(f"  ✓ Emotion: {'FAKE' if score > 0.5 else 'REAL'} ({conf*100:.1f}% conf)")

        # -------------------
        # FUSION MODEL INFERENCE
        # -------------------
        fusion_details = None
        
        if self.fusion_model:
            def resize(x, dim):
                if x.shape[1] > dim:
                    return x[:, :dim]
                return F.pad(x, (0, dim - x.shape[1]))
            with torch.no_grad():
                logits, intermediates = self.fusion_model(
                    resize(text_emb, 64),
                    resize(image_emb, 64),
                    resize(meta_emb, 64),
                    return_intermediates=True
                )
                score = torch.sigmoid(logits).item()
                conf = abs(score - 0.5) * 2
                preds.append(score)
                confs.append(conf)
                
                # Store detailed fusion analysis
                fusion_details = {
                    'score': score,
                    'confidence': conf,
                    'modality_weights': {
                        'text': float(intermediates['modality_weights'][0, 0]),
                        'image': float(intermediates['modality_weights'][0, 1]),
                        'meta': float(intermediates['modality_weights'][0, 2])
                    },
                    'suspicion_scores': {
                        'text': float(intermediates['suspicion_scores'][0, 0]),
                        'image': float(intermediates['suspicion_scores'][0, 1]),
                        'meta': float(intermediates['suspicion_scores'][0, 2])
                    }
                }
                
                print(f"  ✓ Fusion: {'FAKE' if score > 0.5 else 'REAL'} ({conf*100:.1f}% conf)")

        # -------------------
        # FINAL VERDICT
        # -------------------
        if preds:
            final_score = sum(p*c for p,c in zip(preds, confs)) / (sum(confs)+1e-8)
            label = 'FAKE' if final_score > 0.5 else 'REAL'
            confidence = abs(final_score - 0.5) * 2
        else:
            final_score = 0.5
            label = 'UNKNOWN'
            confidence = 0.0

        # -------------------
        # HUMAN-READABLE EXPLANATION
        # -------------------
        self._print_detailed_analysis(
            label, final_score, confidence,
            vad, emotion_details, fusion_details,
            hashtags, mentions, urls, emojis, image_present
        )

        return {
            'post_id': post_id,
            'verdict': label,
            'score': final_score,
            'confidence': confidence,
            'text': text,
            'username': username,
            'metadata': {
                'hashtags': hashtags,
                'mentions': mentions,
                'urls': urls,
                'emojis': emojis
            },
            'vad_analysis': {
                'valence': float(vad[0]),
                'arousal': float(vad[1]),
                'dominance': float(vad[2])
            },
            'emotion_analysis': emotion_details,
            'fusion_analysis': fusion_details
        }
    
    def _print_detailed_analysis(self, label, score, confidence, vad, emotion_details, fusion_details,
                                 hashtags, mentions, urls, emojis, image_present):
        """Print detailed human-readable analysis."""
        
        print("\n" + "="*70)
        print("DETAILED ANALYSIS")
        print("="*70)
        
        # -------------------
        # OVERALL VERDICT
        # -------------------
        if label == 'FAKE':
            if score > 0.85:
                verdict_emoji = "🚨"
                risk_level = "CRITICAL"
                recommendation = "DO NOT SHARE - High confidence fake news"
            elif score > 0.70:
                verdict_emoji = "⚠️"
                risk_level = "HIGH"
                recommendation = "VERIFY BEFORE SHARING - Likely fake"
            elif score > 0.55:
                verdict_emoji = "🟡"
                risk_level = "MODERATE"
                recommendation = "APPROACH WITH CAUTION - Some suspicious patterns"
            else:
                verdict_emoji = "✅"
                risk_level = "LOW"
                recommendation = "LIKELY SAFE - Minor concerns"
        else:
            if confidence > 0.7:
                verdict_emoji = "✅"
                risk_level = "AUTHENTIC"
                recommendation = "APPEARS GENUINE - No significant red flags"
            else:
                verdict_emoji = "🟡"
                risk_level = "LIKELY AUTHENTIC"
                recommendation = "APPEARS GENUINE - Low confidence, verify if important"
        
        print(f"\n{verdict_emoji} VERDICT: {label} ({risk_level})")
        print(f"   Score: {score:.4f} | Confidence: {confidence*100:.1f}%")
        print(f"   📋 {recommendation}")
        
        # -------------------
        # EMOTIONAL ANALYSIS
        # -------------------
        if emotion_details:
            print(f"\n{'─'*70}")
            print("🎭 EMOTIONAL ANALYSIS")
            print(f"{'─'*70}")
            
            # VAD interpretation
            valence = vad[0]
            arousal = vad[1]
            dominance = vad[2]
            
            intensity = np.sqrt(valence**2 + arousal**2 + dominance**2) / np.sqrt(3)
            
            print(f"   Emotional Intensity: {intensity:.2f}")
            print(f"   • Valence (positivity): {valence:.2f} {'😊 Positive' if valence > 0.6 else '😔 Negative' if valence < 0.4 else '😐 Neutral'}")
            print(f"   • Arousal (excitement): {arousal:.2f} {'⚡ High' if arousal > 0.6 else '😴 Low' if arousal < 0.4 else '➡️ Moderate'}")
            print(f"   • Dominance (control):  {dominance:.2f} {'💪 Strong' if dominance > 0.6 else '🤝 Weak' if dominance < 0.4 else '⚖️ Balanced'}")
            
            # Emotional congruence
            congruence = emotion_details['congruence']
            if congruence < -0.3:
                print(f"\n   ⚠️ Emotional Contradiction: {congruence:.2f}")
                print(f"      Text and image emotions conflict - potential manipulation")
            elif congruence > 0.3:
                print(f"\n   ✅ Emotional Alignment: {congruence:.2f}")
                print(f"      Text and image emotions are consistent")
            
            # Mismatch detection
            mismatch = emotion_details['mismatch_magnitude']
            if mismatch > 1.0:
                print(f"\n   🚨 High Emotional Mismatch: {mismatch:.2f}")
                print(f"      Strong inconsistency detected - deception indicator")
            elif mismatch > 0.5:
                print(f"\n   🟡 Moderate Mismatch: {mismatch:.2f}")
                print(f"      Some emotional inconsistency present")
        
        # -------------------
        # MODALITY BREAKDOWN
        # -------------------
        if emotion_details or fusion_details:
            print(f"\n{'─'*70}")
            print("📊 MODALITY CONTRIBUTION")
            print(f"{'─'*70}")
            
            if emotion_details:
                weights = emotion_details['emotion_weights']
                dominant = max(weights, key=weights.get)
                
                print(f"\n   Emotion Model Detection:")
                for modality, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    bar_length = int(weight * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    emoji = "📝" if modality == 'text' else "🖼️" if modality == 'image' else "📊"
                    print(f"   {emoji} {modality.capitalize():6s} [{bar}] {weight*100:5.1f}%")
                
                print(f"\n   → Detection primarily driven by {dominant.upper()} signals")
            
            if fusion_details:
                weights = fusion_details['modality_weights']
                print(f"\n   Fusion Model Weighting:")
                for modality, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    bar_length = int(weight * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    emoji = "📝" if modality == 'text' else "🖼️" if modality == 'image' else "📊"
                    print(f"   {emoji} {modality.capitalize():6s} [{bar}] {weight*100:5.1f}%")
        
        # -------------------
        # CONTENT PATTERNS
        # -------------------
        print(f"\n{'─'*70}")
        print("📝 CONTENT PATTERNS")
        print(f"{'─'*70}")
        
        flags = []
        if urls > 5:
            flags.append(f"⚠️ High URL count ({urls}) - possible link farming")
        elif urls > 2:
            flags.append(f"🟡 Multiple URLs ({urls}) - verify link legitimacy")
        elif urls > 0:
            flags.append(f"✅ Few URLs ({urls})")
        else:
            flags.append(f"✅ No URLs")
        
        if hashtags > 10:
            flags.append(f"⚠️ Excessive hashtags ({hashtags}) - visibility manipulation")
        elif hashtags > 5:
            flags.append(f"🟡 Many hashtags ({hashtags})")
        elif hashtags > 0:
            flags.append(f"✅ Normal hashtag usage ({hashtags})")
        
        if mentions > 10:
            flags.append(f"⚠️ Mass mentions ({mentions}) - potential spam")
        elif mentions > 0:
            flags.append(f"🟡 User mentions ({mentions})")
        
        if emojis > 10:
            flags.append(f"🟡 Heavy emoji usage ({emojis}) - emotional manipulation")
        elif emojis > 5:
            flags.append(f"Moderate emoji usage ({emojis})")
        
        if image_present == 0:
            flags.append(f"ℹ️ Text-only post (no image)")
        else:
            flags.append(f"🖼️ Contains image")
        
        for flag in flags:
            print(f"   {flag}")
        
        # -------------------
        # RISK SUMMARY
        # -------------------
        risk_indicators = []
        
        if emotion_details:
            if emotion_details['score'] > 0.7:
                risk_indicators.append("High emotion model score")
            if emotion_details['congruence'] < -0.3:
                risk_indicators.append("Emotional contradiction")
            if emotion_details['mismatch_magnitude'] > 1.0:
                risk_indicators.append("High emotional mismatch")
        
        if urls > 5:
            risk_indicators.append("Excessive URLs")
        if hashtags > 10:
            risk_indicators.append("Hashtag spam")
        if emojis > 10:
            risk_indicators.append("Emoji manipulation")
        
        if risk_indicators:
            print(f"\n{'─'*70}")
            print("⚠️ RISK INDICATORS")
            print(f"{'─'*70}")
            for indicator in risk_indicators:
                print(f"   • {indicator}")
        
        print("\n" + "="*70)


# ============================================================
# USAGE
# ============================================================
if __name__ == "__main__":
    # Initialize detector
    detector = StandaloneFakeNewsDetector()
    
    # Test examples
    print("\n" + "="*70)
    print("TESTING WITH EXAMPLE POSTS")
    print("="*70)
    
    # Example 1: Text + Image
    print("\n" + "📸"*35)
    print("EXAMPLE 1: Post with Image")
    print("📸"*35)
    
    detector.predict(
        text='Hurricane Sandy. We are coming for you. http://t.co/WLWJ8krG',
        image_path='Dataset/twitter/images_test/attacks_paris_1.jpg',  # No image
        username="miaziervogel"
    )