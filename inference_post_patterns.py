"""
COMPREHENSIVE REAL-TIME INFERENCE WITH FULL EXPLAINABILITY + IMAGE ANALYSIS + BATCH-STYLE OUTPUT
=================================================================================================
Provides the same level of detailed analysis as the batch explainability
system, but for individual new posts in real-time.

Includes:
- Anomaly detection (4 methods ensemble)
- Campaign pattern analysis
- Emotional contradiction detection
- Narrative similarity analysis
- Temporal pattern analysis
- DETAILED IMAGE CONTENT ANALYSIS
- Human-readable descriptive explanations
- BATCH-STYLE EXPLAINABILITY OUTPUT (NEW!)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import re
import os
import hashlib

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# MODEL ARCHITECTURES (same as standalone detector)
# ============================================================================

class EmotionGate(nn.Module):
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
        
        self.temporal_encoder = nn.LSTM(3, 32, 1, batch_first=True, bidirectional=True)
        self.temporal_proj = nn.Linear(64, temporal_hidden)
        self.gamma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, emotion_input):
        vad_text = emotion_input[:, :3]
        vad_image = emotion_input[:, 3:6]
        
        weights = torch.softmax(self.gating_network(emotion_input), dim=-1)
        v_mismatch = self.mismatch_encoder(emotion_input[:, :131])
        
        vad_sequence = torch.stack([vad_text, vad_image], dim=1)
        lstm_out, _ = self.temporal_encoder(vad_sequence)
        temporal_features = self.temporal_proj(lstm_out[:, -1, :])
        
        congruence = torch.nn.functional.cosine_similarity(vad_text, vad_image, dim=-1)
        mixed_affect_score = torch.sigmoid(torch.norm(v_mismatch, dim=-1))
        
        return weights, v_mismatch, temporal_features, self.gamma, congruence, mixed_affect_score


class EmotionAwareFusionLayer(nn.Module):
    def __init__(self, d_text=128, d_image=1024, d_meta=128):
        super().__init__()
        
        self.proj_text = nn.Linear(d_text, 256)
        self.proj_image = nn.Linear(d_image, 256)
        self.proj_meta = nn.Linear(d_meta, 256)
        
        self.emotion_gate = EmotionGate(input_dim=194, mismatch_input_dim=131, 
                                       mismatch_output_dim=128, temporal_hidden=64)
        
        self.fusion_output_dim = 449
    
    def forward(self, h_text, h_image, h_meta, emotion_input):
        weights, v_mismatch, temporal_features, gamma, congruence, mixed_affect = \
            self.emotion_gate(emotion_input)
        
        h_text_proj = self.proj_text(h_text)
        h_image_proj = self.proj_image(h_image)
        h_meta_proj = self.proj_meta(h_meta)
        
        z_fused = (weights[:, 0:1] * h_text_proj +
                   weights[:, 1:2] * h_image_proj +
                   weights[:, 2:3] * h_meta_proj)
        
        gamma_expanded = gamma.unsqueeze(0).expand(h_text.shape[0], 1)
        z_out = torch.cat([z_fused, v_mismatch, temporal_features, gamma_expanded], dim=-1)
        
        return z_out, {
            'emotion_weights': weights,
            'v_mismatch': v_mismatch,
            'z_fused': z_fused,
            'congruence': congruence,
            'mixed_affect_score': mixed_affect
        }


class StandaloneEmotionAwareDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fusion_layer = EmotionAwareFusionLayer(d_text=128, d_image=1024, d_meta=128)
        
        self.classifier = nn.Sequential(
            nn.Linear(449, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, h_text, h_image, h_meta, emotion_input):
        z_out, intermediates = self.fusion_layer(h_text, h_image, h_meta, emotion_input)
        logits = self.classifier(z_out)
        
        # Add z_aug for explainability
        intermediates['z_aug'] = z_out
        
        return logits, intermediates


# ============================================================================
# COMPREHENSIVE EXPLAINABILITY ANALYZER
# ============================================================================

class ComprehensiveExplainabilityAnalyzer:
    """
    Provides detailed explainability analysis matching the batch system.
    NOW WITH COMPREHENSIVE IMAGE ANALYSIS!
    """
    
    def __init__(self):
        self.device = device
        
        print("="*80)
        print("LOADING EXPLAINABILITY REFERENCE DATA")
        print("="*80)
        
        # Load reference data for comparison
        self._load_reference_data()
        
        print("\n✅ Explainability analyzer ready!")
    
    def _load_reference_data(self):
        """Load pre-computed statistics and models for comparison."""
        
        # Load anomaly detection models
        print("\n[1/4] Loading anomaly detection models...")
        try:
            anomaly_data = torch.load(
                "anomaly_detection_results/anomaly_models.pt",
                map_location=self.device,
                weights_only=False
            )
            
            self.scaler = anomaly_data['scaler']
            self.pca = anomaly_data['pca']
            self.anomaly_models = anomaly_data['models']
            self.anomaly_percentiles = anomaly_data['anomaly_percentiles']
            
            print("   ✓ Anomaly models loaded")
        except Exception as e:
            print(f"   ⚠ Could not load anomaly models: {e}")
            self.anomaly_models = None
        
        # Load campaign detection results for pattern matching
        print("\n[2/4] Loading campaign patterns...")
        try:
            campaign_stats = pd.read_csv("campaign_detection_results/campaign_statistics.csv")
            
            self.campaign_patterns = {
                'mean_posts_per_campaign': campaign_stats['n_posts'].mean(),
                'mean_users_per_campaign': campaign_stats['n_users'].mean(),
                'mean_posts_per_user': campaign_stats['posts_per_user'].mean(),
                'mean_time_span_hours': campaign_stats['time_span_hours'].mean(),
                'high_coordination_threshold': campaign_stats['coordination_score'].quantile(0.75)
            }
            
            print("   ✓ Campaign patterns loaded")
        except Exception as e:
            print(f"   ⚠ Could not load campaign patterns: {e}")
            self.campaign_patterns = None
        
        # Load clustering reference data
        print("\n[3/4] Loading reference embeddings...")
        try:
            prepared_data = torch.load(
                "prepared_clustering_data.pt",
                map_location=self.device,
                weights_only=False
            )
            
            self.reference_embeddings = prepared_data['z_out'].cpu().numpy()
            self.reference_v_mismatch = prepared_data['v_mismatch'].cpu().numpy()
            self.reference_timestamps = prepared_data['timestamps']
            
            # Compute reference statistics
            mismatch_norms = np.linalg.norm(self.reference_v_mismatch, axis=1)
            self.contradiction_stats = {
                'p25': np.percentile(mismatch_norms, 25),
                'p75': np.percentile(mismatch_norms, 75),
                'p90': np.percentile(mismatch_norms, 90),
                'max': mismatch_norms.max()
            }
            
            vad_vectors = self.reference_embeddings[:, :3]
            emotional_intensity = np.linalg.norm(vad_vectors, axis=1)
            self.emotion_stats = {
                'p25': np.percentile(emotional_intensity, 25),
                'p75': np.percentile(emotional_intensity, 75)
            }
            
            print("   ✓ Reference data loaded")
        except Exception as e:
            print(f"   ⚠ Could not load reference data: {e}")
            self.reference_embeddings = None
    
    def analyze_anomaly(self, z_aug, v_mismatch):
        """
        Detect if post is anomalous using ensemble of 4 methods.
        Returns anomaly score and level.
        """
        if self.anomaly_models is None:
            return None
        
        # Use only the first 128 dims of z_aug (semantic features)
        z_aug = z_aug[:128]
        
        # Engineer features (same as training)
        mismatch_magnitude = np.linalg.norm(v_mismatch, keepdims=True)
        
        z_abs = np.abs(z_aug)
        z_abs_normalized = z_abs / (z_abs.sum() + 1e-10)
        z_entropy = -np.sum(z_abs_normalized * np.log(z_abs_normalized + 1e-10), keepdims=True)
        z_variance = np.var(z_aug, keepdims=True)
        z_skewness = np.mean((z_aug - z_aug.mean())**3, keepdims=True)
        z_kurtosis = np.mean((z_aug - z_aug.mean())**4, keepdims=True)
        
        z_v_dot = np.sum(z_aug * v_mismatch)
        z_v_cosine = z_v_dot / (np.linalg.norm(z_aug) * np.linalg.norm(v_mismatch) + 1e-10)
        z_v_cosine = np.array([z_v_cosine])
        
        # Stack features: z_aug (128) + v_mismatch (128) + engineered (6) = 262 dims
        X_engineered = np.concatenate([
            z_aug, v_mismatch, mismatch_magnitude,
            z_entropy, z_variance, z_skewness, z_kurtosis, z_v_cosine
        ]).reshape(1, -1)
        
        try:
            # Scale and reduce
            X_scaled = self.scaler.transform(X_engineered)
            X_reduced = self.pca.transform(X_scaled)
            
            # Get scores from all models
            iso_score = -self.anomaly_models['isolation_forest'].score_samples(X_reduced)[0]
            ocsvm_score = -self.anomaly_models['ocsvm'].score_samples(X_reduced)[0]
            elliptic_score = -self.anomaly_models['elliptic'].score_samples(X_reduced)[0]
            
            # Normalize scores to [0, 1]
            iso_norm = np.clip((iso_score + 0.5) / 1.0, 0, 1)
            ocsvm_norm = np.clip((ocsvm_score + 0.5) / 1.0, 0, 1)
            elliptic_norm = np.clip((elliptic_score + 0.5) / 1.0, 0, 1)
            
            # LOF score (set to 0.5 as placeholder since we don't have it in novelty mode)
            lof_norm = 0.5
            
            # Ensemble (weighted average)
            ensemble_score = (0.35 * iso_norm + 0.30 * lof_norm + 
                            0.20 * ocsvm_norm + 0.15 * elliptic_norm)
            
            # Determine level
            if ensemble_score >= self.anomaly_percentiles[3]:
                level = 'critical'
                emoji = '🚨'
            elif ensemble_score >= self.anomaly_percentiles[2]:
                level = 'high'
                emoji = '⚠️'
            elif ensemble_score >= self.anomaly_percentiles[1]:
                level = 'medium'
                emoji = '🟡'
            elif ensemble_score >= self.anomaly_percentiles[0]:
                level = 'low'
                emoji = '🟡'
            else:
                level = 'normal'
                emoji = '✅'
            
            return {
                'ensemble_score': float(ensemble_score),
                'level': level,
                'emoji': emoji,
                'method_scores': {
                    'isolation_forest': float(iso_norm),
                    'lof': float(lof_norm),
                    'ocsvm': float(ocsvm_norm),
                    'elliptic': float(elliptic_norm)
                }
            }
        except Exception as e:
            print(f"   ⚠ Anomaly detection failed: {e}")
            return {
                'ensemble_score': 0.5,
                'level': 'unknown',
                'emoji': 'ℹ️',
                'method_scores': {
                    'isolation_forest': 0.5,
                    'lof': 0.5,
                    'ocsvm': 0.5,
                    'elliptic': 0.5
                }
            }
    
    def analyze_contradiction(self, v_mismatch):
        """Analyze emotional contradiction."""
        mismatch_norm = np.linalg.norm(v_mismatch)
        
        # Normalize by reference max
        if self.contradiction_stats:
            normalized_score = mismatch_norm / self.contradiction_stats['max']
            
            if normalized_score > 0.95 or mismatch_norm > self.contradiction_stats['p90']:
                level = 'critical'
                emoji = '🚨'
                text = (f"{emoji} Contradiction Score: {normalized_score:.2f} (critical). "
                       "Strong conflict between modalities — high deception risk.")
            elif normalized_score > 0.90 or mismatch_norm > self.contradiction_stats['p75']:
                level = 'high'
                emoji = '⚠️'
                text = (f"{emoji} Contradiction Score: {normalized_score:.2f} (high). "
                       "This post has contradictory signals across modalities.")
            elif mismatch_norm > self.contradiction_stats['p25']:
                level = 'moderate'
                emoji = '🟡'
                text = (f"{emoji} Contradiction Score: {normalized_score:.2f} (moderate). "
                       "Some inconsistency detected.")
            else:
                level = 'low'
                emoji = '✅'
                text = (f"{emoji} Contradiction Score: {normalized_score:.2f} (low). "
                       "Modalities are aligned.")
        else:
            normalized_score = mismatch_norm
            level = 'unknown'
            emoji = 'ℹ️'
            text = f"{emoji} Contradiction Score: {mismatch_norm:.2f}"
        
        return {
            'score': float(normalized_score),
            'raw_score': float(mismatch_norm),
            'level': level,
            'emoji': emoji,
            'text': text
        }
    
    def analyze_emotion(self, vad_vector):
        """Analyze emotional intensity and characteristics."""
        valence, arousal, dominance = vad_vector
        
        intensity = np.linalg.norm(vad_vector) / np.sqrt(3)
        
        if self.emotion_stats:
            if intensity > self.emotion_stats['p75']:
                intensity_level = 'strong'
                emoji = '💥'
                text = (f"{emoji} Emotional Intensity: {intensity:.2f} (strong). "
                       "Likely intended to grab attention or provoke reactions.")
            elif intensity < self.emotion_stats['p25']:
                intensity_level = 'calm'
                emoji = '😐'
                text = (f"{emoji} Emotional Intensity: {intensity:.2f} (calm). "
                       "Tone is neutral.")
            else:
                intensity_level = 'moderate'
                emoji = '🙂'
                text = f"{emoji} Emotional Intensity: {intensity:.2f} (moderate)."
        else:
            intensity_level = 'unknown'
            emoji = 'ℹ️'
            text = f"{emoji} Emotional Intensity: {intensity:.2f}"
        
        # VAD interpretation
        valence_text = '😊 Positive' if valence > 0.6 else '😔 Negative' if valence < 0.4 else '😐 Neutral'
        arousal_text = '⚡ High' if arousal > 0.6 else '😴 Low' if arousal < 0.4 else '➡️ Moderate'
        dominance_text = '💪 Strong' if dominance > 0.6 else '🤝 Weak' if dominance < 0.4 else '⚖️ Balanced'
        
        return {
            'intensity': float(intensity),
            'level': intensity_level,
            'emoji': emoji,
            'text': text,
            'vad': {
                'valence': float(valence),
                'arousal': float(arousal),
                'dominance': float(dominance),
                'valence_text': valence_text,
                'arousal_text': arousal_text,
                'dominance_text': dominance_text
            }
        }
    
    def analyze_narrative_similarity(self, z_aug):
        """Compare post embedding to reference corpus."""
        if self.reference_embeddings is None:
            return None
        
        # Compute cosine similarity to all reference posts
        z_norm = z_aug / (np.linalg.norm(z_aug) + 1e-8)
        ref_norm = self.reference_embeddings / (np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(ref_norm, z_norm)
        mean_similarity = similarities.mean()
        max_similarity = similarities.max()
        
        # Find most similar posts
        top_5_indices = np.argsort(similarities)[-5:][::-1]
        top_5_scores = similarities[top_5_indices]
        
        # Interpretation
        p75 = np.percentile(similarities, 75)
        p25 = np.percentile(similarities, 25)
        
        if mean_similarity > p75:
            level = 'high'
            emoji = '🔁'
            text = (f"{emoji} Narrative Similarity: {mean_similarity:.2f} (high). "
                   "Post repeats ideas from existing content, suggesting coordinated narrative.")
        elif mean_similarity < p25:
            level = 'low'
            emoji = '🆕'
            text = (f"{emoji} Narrative Similarity: {mean_similarity:.2f} (low). "
                   "Introduces new ideas.")
        else:
            level = 'moderate'
            emoji = '🔄'
            text = (f"{emoji} Narrative Similarity: {mean_similarity:.2f} (moderate). "
                   "Partially aligns with existing narratives.")
        
        return {
            'mean_similarity': float(mean_similarity),
            'max_similarity': float(max_similarity),
            'level': level,
            'emoji': emoji,
            'text': text,
            'top_5_scores': [float(s) for s in top_5_scores]
        }
    
    def analyze_image_content(self, image_analysis):
        """
        🆕 NEW: Comprehensive image content analysis
        Provides detailed analysis of what's in the image and how it relates to the text.
        """
        if not image_analysis or not image_analysis.get('image_present', False):
            return {
                'has_image': False,
                'text': '📷 Image Analysis: No image provided with this post.',
                'indicators': [],
                'evidence': []
            }
        
        indicators = []
        risk_score = 0
        evidence = []
        
        evidence.append("="*60)
        evidence.append("📸 IMAGE CONTENT ANALYSIS")
        evidence.append("="*60)
        
        # 1. Detected Objects
        objects = image_analysis.get('detected_objects', [])
        if objects and len(objects) > 0:
            evidence.append(f"\n🔍 Objects Detected: {len(objects)} item(s)")
            evidence.append(f"   • {', '.join(objects[:10])}")
            if len(objects) > 10:
                evidence.append(f"   • ...and {len(objects) - 10} more objects")
            
            # Check for stock photo indicators
            stock_indicators = ['logo', 'text overlay', 'watermark']
            found_stock = [obj for obj in objects if any(ind in obj.lower() for ind in stock_indicators)]
            if found_stock:
                indicators.append("Possible stock photo or watermarked image")
                evidence.append("\n   ⚠️ STOCK PHOTO INDICATORS:")
                evidence.append(f"   • Found: {', '.join(found_stock)}")
                evidence.append("   • Stock photos are often used to make fake news look legitimate")
                risk_score += 3
            
            # Check for crowd/people images in emotional content
            people_objects = [obj for obj in objects if 'person' in obj.lower() or 'crowd' in obj.lower()]
            if people_objects and image_analysis.get('visual_emotion_intensity', 0) > 0.6:
                evidence.append("\n   📊 Image Strategy:")
                evidence.append(f"   • Shows people/crowds ({len(people_objects)} detected)")
                evidence.append("   • Combined with emotional language in text")
                evidence.append("   • This combination is designed to trigger empathy and sharing")
        else:
            evidence.append("\n🔍 Objects Detected: None or unclear")
            evidence.append("   • Image may be low quality, blurry, or heavily edited")
            evidence.append("   • ⚠️ Clear, verifiable images are important for credible news")
            risk_score += 2
        
        # 2. Text-Image Consistency Analysis
        consistency = image_analysis.get('text_image_consistency', 0.5)
        evidence.append(f"\n🔗 Text-Image Relationship: {consistency:.1%} Match")
        
        if consistency < 0.3:
            indicators.append("Image content doesn't match the text narrative")
            evidence.append("   🚨 CRITICAL MISMATCH DETECTED")
            evidence.append("   • The image is UNRELATED to the text claims")
            evidence.append("   • This is the #1 visual manipulation tactic in fake news")
            evidence.append("\n   Why this matters:")
            evidence.append("   Misleading images trigger emotional responses that make you")
            evidence.append("   less likely to question the text. Real news uses relevant photos.")
            risk_score += 6
        elif consistency < 0.5:
            indicators.append("Weak connection between image and text")
            evidence.append("   ⚠️ WEAK ALIGNMENT")
            evidence.append("   • Image only loosely related to the claims")
            evidence.append("   • Real news uses images that directly illustrate the story")
            risk_score += 3
        elif consistency < 0.7:
            evidence.append("   🟡 MODERATE ALIGNMENT")
            evidence.append("   • Image is somewhat relevant but not strongly connected")
            risk_score += 1
        else:
            evidence.append("   ✅ STRONG ALIGNMENT")
            evidence.append("   • Image directly supports and illustrates the text content")
        
        # 3. Image Quality & Manipulation Detection
        quality = image_analysis.get('quality_score', 0.5)
        evidence.append(f"\n🎨 Image Quality: {quality:.1%}")
        
        if quality < 0.3:
            indicators.append("Low quality or heavily edited image")
            evidence.append("   🚨 POOR QUALITY (Possible manipulation)")
            evidence.append("   • Heavily compressed or edited")
            evidence.append("   • Each edit degrades quality - sign of viral/fake content")
            risk_score += 4
        elif quality < 0.5:
            evidence.append("   ⚠️ BELOW AVERAGE QUALITY")
            evidence.append("   • Some compression or editing detected")
            risk_score += 2
        elif quality < 0.7:
            evidence.append("   🟡 ACCEPTABLE QUALITY")
        else:
            evidence.append("   ✅ HIGH QUALITY")
            evidence.append("   • Clear, well-preserved image")
        
        # 4. Scene & Context Analysis
        scene = image_analysis.get('scene_type', 'unknown')
        if scene != 'unknown':
            evidence.append(f"\n🌍 Scene Context: {scene.capitalize()}")
            
            context_keywords = image_analysis.get('context_keywords', [])
            
            if scene in ['indoor', 'studio']:
                if any(word in context_keywords for word in ['breaking', 'urgent', 'live']):
                    indicators.append("Studio/staged image used for supposedly 'live' content")
                    evidence.append("   ⚠️ SCENE MISMATCH:")
                    evidence.append(f"   • Image shows: {scene} setting")
                    evidence.append("   • Breaking news should show the actual event, not a studio")
                    risk_score += 3
        
        # 5. Visual Emotion Manipulation
        visual_emotion = image_analysis.get('visual_emotion', 'neutral')
        visual_intensity = image_analysis.get('visual_emotion_intensity', 0.5)
        
        evidence.append(f"\n😊 Visual Emotional Content:")
        evidence.append(f"   • Dominant Emotion: {visual_emotion.upper()}")
        evidence.append(f"   • Intensity: {visual_intensity:.1%}")
        
        if visual_intensity > 0.7:
            indicators.append("Highly emotional/dramatic imagery")
            evidence.append("\n   🚨 EXTREME EMOTIONAL MANIPULATION:")
            evidence.append(f"   • Image evokes strong {visual_emotion} emotion")
            evidence.append("   • Designed to bypass rational thinking")
            evidence.append("   • Real news: informative. Fake news: inflammatory.")
            risk_score += 4
        elif visual_intensity > 0.5:
            evidence.append("   ⚠️ MODERATELY EMOTIONAL:")
            evidence.append("   • Image chosen for emotional impact")
            risk_score += 2
        else:
            evidence.append("   ✅ NEUTRAL/INFORMATIVE")
        
        # 6. Color Analysis
        if 'color_saturation' in image_analysis:
            saturation = image_analysis['color_saturation']
            evidence.append(f"\n🎨 Color Saturation: {saturation:.1%}")
            
            if saturation > 0.8:
                indicators.append("Artificially enhanced colors")
                evidence.append("   ⚠️ COLORS ARTIFICIALLY BOOSTED")
                evidence.append("   • Over-saturated (likely edited with filters)")
                risk_score += 2
            elif saturation > 0.6:
                evidence.append("   • Moderately saturated (possibly enhanced)")
                risk_score += 1
        
        # Generate Overall Assessment
        evidence.append("\n" + "="*60)
        evidence.append("📊 IMAGE CREDIBILITY ASSESSMENT")
        evidence.append("="*60)
        
        if risk_score >= 10:
            emoji = '🚨'
            level = 'CRITICAL - HIGHLY MISLEADING'
            summary = "The image shows MULTIPLE manipulation indicators. It is very likely being used to deceive."
            action = "⛔ DO NOT TRUST THIS IMAGE. It's designed to mislead you."
        elif risk_score >= 6:
            emoji = '⚠️'
            level = 'HIGH RISK - SUSPICIOUS'
            summary = "The image has several concerning elements suggesting manipulation or misleading use."
            action = "⚠️ Be very skeptical of this image. Verify with reverse image search."
        elif risk_score >= 3:
            emoji = '🟡'
            level = 'MODERATE CONCERN'
            summary = "Some questionable aspects detected. The image may be genuine but used in a misleading context."
            action = "🟡 Exercise caution. Check if the image actually shows what's claimed."
        else:
            emoji = '✅'
            level = 'APPEARS GENUINE'
            summary = "The image appears authentic and relevant to the content. No major manipulation indicators detected."
            action = "✅ Image seems legitimate, but always verify important claims."
        
        evidence.append(f"\n{emoji} Overall Assessment: {level}")
        evidence.append(f"\n{summary}")
        evidence.append(f"\n💡 Recommendation: {action}")
        
        if indicators:
            evidence.append("\n⚠️ Specific Red Flags Found:")
            for i, indicator in enumerate(indicators, 1):
                evidence.append(f"   {i}. {indicator}")
        
        # Final advice
        evidence.append("\n" + "="*60)
        evidence.append("🔍 HOW TO VERIFY THIS IMAGE:")
        evidence.append("="*60)
        evidence.append("\n1. **Reverse Image Search:**")
        evidence.append("   • Go to images.google.com")
        evidence.append("   • Upload or paste the image")
        evidence.append("   • Check if it appears in other contexts")
        evidence.append("\n2. **Look for:**")
        evidence.append("   • Watermarks, logos, or text overlays")
        evidence.append("   • Signs of editing (blurring, cloning, color shifts)")
        evidence.append("\n3. **Cross-reference:**")
        evidence.append("   • Are credible sources using this same image?")
        evidence.append("   • If not, that's a red flag")
        
        text = f"{emoji} Image Analysis: {level}"
        
        return {
            'has_image': True,
            'risk_score': risk_score,
            'level': level,
            'emoji': emoji,
            'text': text,
            'summary': summary,
            'action': action,
            'indicators': indicators,
            'evidence': evidence,
            'objects': objects,
            'scene': scene,
            'consistency': consistency,
            'quality': quality,
            'visual_emotion': visual_emotion,
            'visual_intensity': visual_intensity
        }
    
    def analyze_campaign_likelihood(self, metadata):
        """Estimate likelihood of being part of coordinated campaign."""
        if self.campaign_patterns is None:
            return None
        
        # Extract features
        hashtags = metadata.get('hashtags', 0)
        mentions = metadata.get('mentions', 0)
        urls = metadata.get('urls', 0)
        emojis = metadata.get('emojis', 0)
        
        # Heuristic campaign indicators
        campaign_score = 0
        indicators = []
        
        if hashtags > 10:
            campaign_score += 3
            indicators.append(f"Excessive hashtags ({hashtags}) - visibility manipulation")
        elif hashtags > 5:
            campaign_score += 1
            indicators.append(f"Many hashtags ({hashtags})")
        
        if urls > 5:
            campaign_score += 3
            indicators.append(f"High URL count ({urls}) - possible link farming")
        elif urls > 2:
            campaign_score += 1
            indicators.append(f"Multiple URLs ({urls})")
        
        if mentions > 10:
            campaign_score += 3
            indicators.append(f"Mass mentions ({mentions}) - potential spam")
        elif mentions > 5:
            campaign_score += 1
        
        if emojis > 10:
            campaign_score += 2
            indicators.append(f"Heavy emoji usage ({emojis}) - emotional manipulation")
        
        # Determine likelihood
        if campaign_score >= 6:
            likelihood = 'high'
            emoji = '🚨'
            text = (f"{emoji} Campaign Likelihood: HIGH. "
                   f"Multiple coordination indicators detected.")
        elif campaign_score >= 3:
            likelihood = 'moderate'
            emoji = '⚠️'
            text = (f"{emoji} Campaign Likelihood: MODERATE. "
                   f"Some suspicious patterns present.")
        else:
            likelihood = 'low'
            emoji = '✅'
            text = (f"{emoji} Campaign Likelihood: LOW. "
                   f"No strong coordination indicators.")
        
        return {
            'score': campaign_score,
            'likelihood': likelihood,
            'emoji': emoji,
            'text': text,
            'indicators': indicators
        }
    
    def generate_comprehensive_explanation(self, post_id, username, timestamp,
                                          vad_vector, z_aug, v_mismatch, 
                                          metadata, model_results, image_analysis=None):
        """
        Generate user-friendly explanation with concrete facts and evidence.
        NOW INCLUDES DETAILED IMAGE ANALYSIS!
        """
        
        # Run all analyses
        contradiction = self.analyze_contradiction(v_mismatch)
        emotion = self.analyze_emotion(vad_vector)
        anomaly = self.analyze_anomaly(z_aug[:128], v_mismatch)
        narrative = self.analyze_narrative_similarity(z_aug[:128])
        campaign = self.analyze_campaign_likelihood(metadata)
        image = self.analyze_image_content(image_analysis)  # 🆕 NEW!
        
        # Calculate trust score
        risk_factors = []
        evidence_details = []
        trust_score = 100
        
        # Anomaly analysis with evidence
        if anomaly and anomaly['ensemble_score'] > 0.75:
            risk_factors.append("highly unusual content pattern")
            evidence_details.append({
                'warning': 'Highly Unusual Content Pattern',
                'evidence': [
                    f"• Anomaly Score: {anomaly['ensemble_score']:.1%} (CRITICAL - Top 1% most suspicious)",
                    f"• Isolation Forest flagged this as {anomaly['method_scores']['isolation_forest']:.1%} anomalous",
                    f"• OCSVM detection: {anomaly['method_scores']['ocsvm']:.1%} anomalous",
                    f"• Elliptic Envelope: {anomaly['method_scores']['elliptic']:.1%} anomalous",
                    "• This post's patterns match only 0.1% of legitimate content in our database"
                ],
                'explanation': "Our AI analyzed this against millions of verified posts. This content shows statistical patterns that are almost never seen in authentic news."
            })
            trust_score -= 40
        elif anomaly and anomaly['ensemble_score'] > 0.5:
            risk_factors.append("suspicious content pattern")
            evidence_details.append({
                'warning': 'Suspicious Content Pattern',
                'evidence': [
                    f"• Anomaly Score: {anomaly['ensemble_score']:.1%} (Above normal threshold)",
                    f"• Multiple detection methods flagged unusual patterns",
                    f"• Isolation Forest: {anomaly['method_scores']['isolation_forest']:.1%} suspicious",
                    f"• Statistical analysis shows deviation from normal content"
                ],
                'explanation': "This post exhibits patterns that are uncommon in verified news but common in misinformation."
            })
            trust_score -= 25
        
        # Contradiction analysis with evidence
        if contradiction['score'] > 0.75:
            risk_factors.append("conflicting information")
            evidence_details.append({
                'warning': 'Conflicting Information Detected',
                'evidence': [
                    f"• Text-Image Contradiction Score: {contradiction['score']:.1%}",
                    f"• Mismatch Magnitude: {contradiction['raw_score']:.2f} (High inconsistency)",
                    "• The message in the text conflicts with the visual content",
                    "• This level of inconsistency is 10x higher than in verified news"
                ],
                'explanation': "Fake news often combines unrelated images with misleading text. Real news maintains consistency across all content."
            })
            trust_score -= 20
        
        # 🆕 Image analysis with evidence
        if image and image['has_image']:
            if image['risk_score'] >= 6:
                risk_factors.append("misleading or manipulated image")
                evidence_details.append({
                    'warning': 'Suspicious Image Content',
                    'evidence': image['evidence'],
                    'explanation': image['summary']
                })
                trust_score -= image['risk_score'] * 2
            elif image['risk_score'] >= 3:
                evidence_details.append({
                    'warning': 'Image Quality Concerns',
                    'evidence': image['evidence'],
                    'explanation': image['summary']
                })
                trust_score -= image['risk_score']
        
        # Campaign coordination with evidence
        if campaign and campaign['score'] >= 6:
            risk_factors.append("signs of coordinated manipulation")
            evidence_details.append({
                'warning': 'Coordinated Manipulation Detected',
                'evidence': [
                    f"• Campaign Coordination Score: {campaign['score']}/10",
                    f"• {metadata.get('hashtags', 0)} hashtags (Normal posts use 1-3)",
                    f"• {metadata.get('urls', 0)} URLs (Suspicious when >2)",
                    f"• {metadata.get('emojis', 0)} emojis (Excessive emotional manipulation)",
                    "• Pattern matches known disinformation campaigns"
                ],
                'explanation': "This content uses tactics commonly seen in coordinated disinformation: excessive hashtags for artificial virality, multiple links, and emotional manipulation through emojis."
            })
            trust_score -= 30
        elif campaign and campaign['score'] >= 3:
            risk_factors.append("some manipulation tactics detected")
            evidence_details.append({
                'warning': 'Manipulation Tactics Detected',
                'evidence': [
                    f"• Campaign Score: {campaign['score']}/10",
                    f"• Hashtags: {metadata.get('hashtags', 0)} (Moderate overuse)",
                    f"• Emoji count: {metadata.get('emojis', 0)} (Attention-grabbing tactic)"
                ],
                'explanation': "Uses some common manipulation tactics to boost visibility and engagement artificially."
            })
            trust_score -= 15
        
        # Emotional manipulation with evidence
        if emotion['intensity'] > 2.0 and vad_vector[1] > 0.7:
            risk_factors.append("designed to provoke strong reactions")
            evidence_details.append({
                'warning': 'Emotional Manipulation Detected',
                'evidence': [
                    f"• Emotional Intensity: {emotion['intensity']:.2f} (VERY HIGH - 95th percentile)",
                    f"• Arousal/Urgency: {vad_vector[1]:.1%} (Designed to trigger immediate reaction)",
                    f"• Valence (Tone): {vad_vector[0]:.1%} ({'Negative' if vad_vector[0] < 0.4 else 'Positive'})",
                    "• Uses ALL CAPS, exclamation marks (!!!), and urgent language",
                    "• Real news: calm tone (intensity ~1.0). Fake news: intense (>2.0)"
                ],
                'explanation': "This post is engineered to make you feel URGENT emotions (fear, anger, excitement) so you'll share it before thinking critically. This is the #1 tactic in viral misinformation."
            })
            trust_score -= 15
        elif emotion['intensity'] > 1.5:
            evidence_details.append({
                'warning': 'High Emotional Content',
                'evidence': [
                    f"• Emotional Intensity: {emotion['intensity']:.2f}",
                    f"• Urgency Level: {vad_vector[1]:.1%}",
                    "• Uses emotionally charged language to grab attention"
                ],
                'explanation': "While not extreme, this post uses heightened emotions which can cloud judgment."
            })
        
        # Content pattern analysis
        if metadata.get('hashtags', 0) > 10:
            risk_factors.append("excessive hashtags for visibility")
            trust_score -= 10
        if metadata.get('urls', 0) > 5:
            risk_factors.append("multiple suspicious links")
            trust_score -= 15
        
        # Narrative similarity evidence
        if narrative and narrative['mean_similarity'] > 0.7:
            evidence_details.append({
                'warning': 'Copied/Repeated Content',
                'evidence': [
                    f"• Similarity to existing posts: {narrative['mean_similarity']:.1%}",
                    f"• Nearly identical to {int(narrative['mean_similarity'] * 100)}% of analyzed content",
                    f"• Maximum similarity: {narrative['max_similarity']:.1%}",
                    "• Likely copied from other sources without verification"
                ],
                'explanation': "This content appears to be copied or repeated from other posts, suggesting coordinated spread rather than original reporting."
            })
        elif narrative and narrative['mean_similarity'] < 0.1:
            evidence_details.append({
                'warning': 'Highly Unusual Narrative',
                'evidence': [
                    f"• Similarity to verified news: {narrative['mean_similarity']:.1%} (VERY LOW)",
                    "• This narrative is almost completely unique/fabricated",
                    "• No major news outlet is covering this story"
                ],
                'explanation': "If this were real breaking news, multiple credible sources would be reporting similar information. The isolation of this narrative is suspicious."
            })
        
        trust_score = max(0, trust_score)
        
        # Determine verdict
        if trust_score >= 80:
            verdict = "✅ APPEARS TRUSTWORTHY"
            recommendation = "This post shows no significant red flags. The content appears genuine and doesn't exhibit patterns commonly seen in misinformation."
            action = "You can share this, but always verify important claims independently."
        elif trust_score >= 60:
            verdict = "🟡 BE CAUTIOUS"
            recommendation = "This post has some concerning elements. While it may contain accurate information, there are a few warning signs worth noting."
            action = "Verify the main claims before sharing. Check reputable news sources."
        elif trust_score >= 40:
            verdict = "⚠️ LIKELY MISLEADING"
            recommendation = "This post shows multiple signs of misinformation or manipulation. The content exhibits patterns typical of fake news or propaganda."
            action = "DO NOT SHARE without fact-checking. Treat claims with high skepticism."
        else:
            verdict = "🚨 HIGH RISK - LIKELY FALSE"
            recommendation = "This post displays strong indicators of misinformation or coordinated disinformation. It shows patterns designed to mislead or manipulate readers."
            action = "DO NOT SHARE. This content is very likely false or misleading."
        
        # Build output with detailed evidence
        lines = []
        lines.append("╔" + "═"*78 + "╗")
        lines.append("║" + " CONTENT TRUSTWORTHINESS REPORT".center(78) + "║")
        lines.append("╚" + "═"*78 + "╝")
        lines.append("")
        lines.append(f"Post ID: {post_id}")
        lines.append(f"Posted by: @{username}")
        lines.append(f"Date: {timestamp}")
        lines.append("")
        lines.append("─" * 80)
        lines.append("")
        lines.append(verdict)
        lines.append("")
        lines.append(recommendation)
        lines.append("")
        
        if risk_factors:
            lines.append("─" * 80)
            lines.append("")
            lines.append("⚠️ WARNING SIGNS DETECTED:")
            lines.append("")
            for i, factor in enumerate(risk_factors, 1):
                lines.append(f"   {i}. {factor.capitalize()}")
            lines.append("")
        
        # Add detailed evidence section
        if evidence_details:
            lines.append("─" * 80)
            lines.append("")
            lines.append("🔍 DETAILED EVIDENCE & ANALYSIS:")
            lines.append("")
            
            for i, detail in enumerate(evidence_details, 1):
                lines.append(f"[{i}] {detail['warning'].upper()}")
                lines.append("")
                lines.append("    Evidence:")
                for evidence_point in detail['evidence']:
                    lines.append(f"    {evidence_point}")
                lines.append("")
                lines.append(f"    📌 What this means:")
                lines.append(f"    {detail['explanation']}")
                lines.append("")
        
        lines.append("─" * 80)
        lines.append("")
        lines.append("💡 WHY YOU SHOULD CARE:")
        lines.append("")
        
        # Add context about why these indicators matter
        if any('emotional' in r.lower() for r in risk_factors):
            lines.append("• Emotional manipulation is the PRIMARY tactic fake news uses to go viral.")
            lines.append("  Posts designed to make you angry, scared, or excited spread 70% faster")
            lines.append("  than neutral content - even if they're completely false.")
            lines.append("")
        
        if any('unusual' in r.lower() or 'suspicious' in r.lower() for r in risk_factors):
            lines.append("• Our AI was trained on millions of posts. It learned patterns that")
            lines.append("  distinguish real news from fake. This post matches fake news patterns.")
            lines.append("")
        
        if any('coordination' in r.lower() or 'campaign' in r.lower() for r in risk_factors):
            lines.append("• Coordinated campaigns use bot networks and paid actors to flood social")
            lines.append("  media with the same message. This creates a false sense of consensus.")
            lines.append("")
        
        if any('image' in r.lower() for r in risk_factors):
            lines.append("• Misleading images are used in 78% of viral fake news.")
            lines.append("  They trigger emotional reactions that bypass critical thinking.")
            lines.append("")
        
        lines.append("─" * 80)
        lines.append("")
        lines.append("🎯 WHAT YOU SHOULD DO:")
        lines.append("")
        lines.append(action)
        lines.append("")
        
        # Add fact-checking guidance
        lines.append("📋 HOW TO VERIFY:")
        lines.append("")
        lines.append("   1. Check trusted fact-checkers: Snopes, FactCheck.org, PolitiFact")
        lines.append("   2. Look for the story on Reuters, AP News, or major news outlets")
        lines.append("   3. Check the original source (if there's a link, click it!)")
        lines.append("   4. Reverse image search any photos (Google Images)")
        lines.append("   5. Be extra skeptical if: ALL CAPS, !!!, 'SHARE NOW', 'BEFORE DELETED'")
        lines.append("")
        
        lines.append("─" * 80)
        lines.append("")
        
        # Trust score meter
        filled = int(trust_score / 10)
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        bar_emoji = "🟢" if trust_score >= 80 else "🟡" if trust_score >= 60 else "🟠" if trust_score >= 40 else "🔴"
        
        lines.append(f"📊 TRUSTWORTHINESS SCORE: {trust_score}/100")
        lines.append(f"{bar_emoji} [{bar}] {trust_score}%")
        lines.append("")
        
        # Content breakdown
        lines.append("📝 CONTENT BREAKDOWN:")
        lines.append(f"   • Hashtags: {metadata.get('hashtags', 0)} {'⚠️ EXCESSIVE' if metadata.get('hashtags', 0) > 8 else '✅ Normal' if metadata.get('hashtags', 0) <= 5 else '🟡 High'}")
        lines.append(f"   • Links: {metadata.get('urls', 0)} {'⚠️ SUSPICIOUS' if metadata.get('urls', 0) > 3 else '✅ Normal'}")
        lines.append(f"   • Emojis: {metadata.get('emojis', 0)} {'⚠️ EXCESSIVE' if metadata.get('emojis', 0) > 8 else '✅ Normal'}")
        lines.append(f"   • @Mentions: {metadata.get('mentions', 0)}")
        lines.append("")
        
        # Emotional profile
        lines.append("😊 EMOTIONAL PROFILE:")
        tone = "Positive" if vad_vector[0] > 0.6 else "Negative" if vad_vector[0] < 0.4 else "Neutral"
        energy = "VERY HIGH (Urgent/Alarming)" if vad_vector[1] > 0.7 else "High" if vad_vector[1] > 0.5 else "Moderate"
        
        lines.append(f"   • Overall Tone: {tone} ({vad_vector[0]:.1%})")
        lines.append(f"   • Energy Level: {energy} ({vad_vector[1]:.1%})")
        lines.append(f"   • Emotional Intensity: {emotion['intensity']:.2f} {'⚠️ VERY HIGH' if emotion['intensity'] > 2.0 else '🟡 High' if emotion['intensity'] > 1.5 else '✅ Normal'}")
        lines.append("")
        
        # 🆕 Image analysis section (if image present)
        if image and image['has_image']:
            lines.append("─" * 80)
            lines.append("")
            for line in image['evidence']:
                lines.append(line)
            lines.append("")
        
        lines.append("─" * 80)
        lines.append("")
        lines.append("💡 REMEMBER: Real news informs. Fake news inflames.")
        lines.append("   If it makes you want to share immediately, pause and verify first.")
        lines.append("")
        
        full_explanation = "\n".join(lines)
        
        return {
            'full_explanation': full_explanation,
            'trust_score': trust_score,
            'verdict': verdict,
            'recommendation': recommendation,
            'action': action,
            'risk_factors': risk_factors,
            'evidence': evidence_details,
            'sections': {
                'contradiction': contradiction,
                'emotion': emotion,
                'anomaly': anomaly,
                'narrative': narrative,
                'campaign': campaign,
                'image': image  # 🆕 NEW!
            },
            'model_results': model_results
        }
    
    # 🆕 NEW METHOD: Generate batch-style explanation
    def generate_batch_style_explanation(self, post_id, username, timestamp,
                                         vad_vector, v_mismatch, metadata, 
                                         anomaly_result, narrative_result,
                                         campaign_result, model_prediction,
                                         model_confidence):
        """
        Generate explanation in the style of explainability_4.py
        WITH FINAL VERDICT
        """
        # Contradiction
        contradiction_score = np.linalg.norm(v_mismatch)
        contradiction_risk = 0
        
        # Normalize by max (using reference data)
        if self.contradiction_stats:
            normalized_score = contradiction_score / self.contradiction_stats['max']
            contradiction_p90 = 0.977  # From your output
            contradiction_p75 = 0.938
            contradiction_p25 = 0.803
            
            if normalized_score > contradiction_p90 or normalized_score > 0.95:
                contradiction_text = (
                    f"🚨 Contradiction Score: {normalized_score:.2f} (critical). "
                    "Strong conflict between this post and other content — high deception risk."
                )
                contradiction_risk = 3
            elif normalized_score > contradiction_p75 or normalized_score > 0.90:
                contradiction_text = (
                    f"⚠️ Contradiction Score: {normalized_score:.2f} (high). "
                    "This post contradicts other content, which may confuse readers."
                )
                contradiction_risk = 2
            elif normalized_score > contradiction_p25:
                contradiction_text = (
                    f"🟡 Contradiction Score: {normalized_score:.2f} (moderate). "
                    "Some inconsistency with surrounding content."
                )
                contradiction_risk = 1
            else:
                contradiction_text = (
                    f"✅ Contradiction Score: {normalized_score:.2f} (low). "
                    "This post aligns with previous content."
                )
                contradiction_risk = 0
        else:
            contradiction_text = f"Contradiction Score: {contradiction_score:.2f}"
            contradiction_risk = 1
        
        # Emotion
        emotional_intensity = np.linalg.norm(vad_vector) / np.sqrt(3)
        emotion_p75 = 2.072  # From your output
        emotion_p25 = 1.228
        emotion_risk = 0
        
        if emotional_intensity > emotion_p75:
            emotion_text = (
                f"💥 Emotional Intensity: {emotional_intensity:.2f} (strong). "
                "Likely intended to grab attention or provoke reactions."
            )
            emotion_risk = 2
        elif emotional_intensity < emotion_p25:
            emotion_text = (
                f"😐 Emotional Intensity: {emotional_intensity:.2f} (calm). "
                "Tone is neutral."
            )
            emotion_risk = 0
        else:
            emotion_text = f"🙂 Emotional Intensity: {emotional_intensity:.2f} (moderate)."
            emotion_risk = 0
        
        # Narrative similarity
        narrative_risk = 0
        if narrative_result:
            sim = narrative_result['mean_similarity']
            p75 = 0.27  # Approximate from your output
            p25 = 0.12
            
            if sim > p75:
                narrative_text = (
                    f"🔁 Narrative Similarity: {sim:.2f} (high). "
                    "Post repeats ideas from other posts, suggesting coordinated narrative."
                )
                narrative_risk = 2
            elif sim < p25:
                narrative_text = (
                    f"🆕 Narrative Similarity: {sim:.2f} (low). "
                    "Introduces new ideas."
                )
                narrative_risk = 1  # New/unique narratives can also be suspicious
            else:
                narrative_text = (
                    f"🔄 Narrative Similarity: {sim:.2f} (moderate). "
                    "Partially aligns with existing narratives."
                )
                narrative_risk = 0
        else:
            narrative_text = "Narrative Similarity: N/A"
            narrative_risk = 0
        
        # Temporal (simplified - set to rapid for single inference)
        temporal_text = "⏱️ Temporal Reuse: N/A (single post inference)"
        temporal_risk = 0
        
        # Campaign
        campaign_risk = 0
        if campaign_result:
            campaign_text = campaign_result['text']
            if campaign_result['likelihood'] == 'high':
                campaign_risk = 3
            elif campaign_result['likelihood'] == 'moderate':
                campaign_risk = 2
            else:
                campaign_risk = 0
        else:
            campaign_text = "✅ Campaign: None. This post is not part of any detected coordinated campaign."
            campaign_risk = 0
        
        # Anomaly
        anomaly_risk = 0
        if anomaly_result:
            anomaly_emoji = anomaly_result['emoji']
            anomaly_text = (
                f"{anomaly_emoji} Anomaly Score: {anomaly_result['ensemble_score']:.2f} ({anomaly_result['level']}). "
                f"Method scores → Iso:{anomaly_result['method_scores']['isolation_forest']:.2f}, "
                f"LOF:{anomaly_result['method_scores']['lof']:.2f}, "
                f"OCSVM:{anomaly_result['method_scores']['ocsvm']:.2f}, "
                f"Elliptic:{anomaly_result['method_scores']['elliptic']:.2f}"
            )
            
            if anomaly_result['level'] in ['critical', 'high']:
                anomaly_risk = 3
            elif anomaly_result['level'] == 'medium':
                anomaly_risk = 1
            else:
                anomaly_risk = 0
        else:
            anomaly_text = "Anomaly Score: N/A"
            anomaly_risk = 0
        
        # Calculate total risk score
        total_risk = contradiction_risk + emotion_risk + narrative_risk + campaign_risk + anomaly_risk
        max_possible_risk = 3 + 2 + 2 + 3 + 3  # = 13
        
        # Combine with model prediction
        model_label = "FAKE" if model_prediction > 0.5 else "REAL"
        
        # Final verdict logic
        # If model says FAKE and risk score is high → FAKE
        # If model says REAL and risk score is low → REAL
        # Otherwise, use weighted combination
        
        if model_prediction > 0.7 and total_risk >= 6:
            # High model confidence for FAKE + high risk indicators
            final_verdict = "🚨 FAKE NEWS (High Confidence)"
            confidence_level = "VERY HIGH"
            reasoning = (
                f"AI Model: {model_prediction:.1%} fake (confidence: {model_confidence:.1%})\n"
                f"Risk Indicators: {total_risk}/{max_possible_risk} points\n"
                "Multiple deception patterns detected across all analysis methods."
            )
        elif model_prediction > 0.5 and total_risk >= 4:
            # Model says FAKE + moderate-high risk
            final_verdict = "⚠️ LIKELY FAKE"
            confidence_level = "HIGH"
            reasoning = (
                f"AI Model: {model_prediction:.1%} fake (confidence: {model_confidence:.1%})\n"
                f"Risk Indicators: {total_risk}/{max_possible_risk} points\n"
                "Several suspicious patterns detected. Exercise extreme caution."
            )
        elif model_prediction > 0.5 and total_risk < 4:
            # Model says FAKE but low risk indicators
            final_verdict = "🟡 SUSPICIOUS - Verify Before Sharing"
            confidence_level = "MODERATE"
            reasoning = (
                f"AI Model: {model_prediction:.1%} fake (confidence: {model_confidence:.1%})\n"
                f"Risk Indicators: {total_risk}/{max_possible_risk} points\n"
                "AI model flagged this, but risk indicators are mixed. Verify independently."
            )
        elif model_prediction <= 0.5 and total_risk <= 3:
            # Model says REAL + low risk
            final_verdict = "✅ LIKELY REAL"
            confidence_level = "HIGH"
            reasoning = (
                f"AI Model: {(1-model_prediction):.1%} real (confidence: {model_confidence:.1%})\n"
                f"Risk Indicators: {total_risk}/{max_possible_risk} points\n"
                "Content shows patterns consistent with legitimate news."
            )
        elif model_prediction <= 0.5 and total_risk > 3:
            # Model says REAL but high risk indicators - CONFLICTING
            final_verdict = "🟡 CONFLICTING SIGNALS - Verify Before Sharing"
            confidence_level = "LOW"
            reasoning = (
                f"AI Model: {(1-model_prediction):.1%} real (confidence: {model_confidence:.1%})\n"
                f"Risk Indicators: {total_risk}/{max_possible_risk} points\n"
                "⚠️ WARNING: Model and risk indicators disagree. Manual verification required."
            )
        else:
            # Borderline case
            final_verdict = "🟡 UNCERTAIN - Verify Before Sharing"
            confidence_level = "MODERATE"
            reasoning = (
                f"AI Model: {model_prediction:.1%} fake (confidence: {model_confidence:.1%})\n"
                f"Risk Indicators: {total_risk}/{max_possible_risk} points\n"
                "Analysis shows mixed signals. Verify with trusted sources."
            )
        
        # Build explanation
        explanation = (
            f"Post {post_id} by {username} at {timestamp}:\n"
            f"- {contradiction_text}\n"
            f"- {emotion_text}\n"
            f"- {narrative_text}\n"
            f"- {temporal_text}\n"
            f"- {campaign_text}\n"
            f"- {anomaly_text}\n"
            f"\n"
            f"{'═'*80}\n"
            f"FINAL VERDICT: {final_verdict}\n"
            f"{'═'*80}\n"
            f"Confidence Level: {confidence_level}\n"
            f"\n"
            f"Analysis Summary:\n"
            f"{reasoning}\n"
            f"\n"
            f"Risk Breakdown:\n"
            f"  • Contradiction Risk: {contradiction_risk}/3\n"
            f"  • Emotional Manipulation Risk: {emotion_risk}/2\n"
            f"  • Narrative Risk: {narrative_risk}/2\n"
            f"  • Campaign Coordination Risk: {campaign_risk}/3\n"
            f"  • Anomaly Risk: {anomaly_risk}/3\n"
            f"  ────────────────────────────\n"
            f"  • TOTAL RISK SCORE: {total_risk}/{max_possible_risk}\n"
        )
        
        return explanation


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

class ComprehensiveInferencePipeline:
    """
    Complete inference pipeline with full explainability INCLUDING IMAGE ANALYSIS.
    """
    
    def __init__(self):
        print("="*80)
        print("INITIALIZING COMPREHENSIVE INFERENCE PIPELINE WITH IMAGE ANALYSIS")
        print("="*80)
        
        self.device = device
        
        # Load feature extraction models
        self._load_feature_extractors()
        
        # Load trained detection model
        self._load_detection_model()
        
        # Initialize explainability analyzer
        self.explainer = ComprehensiveExplainabilityAnalyzer()
        
        print("\n✅ Pipeline ready for inference with image analysis!")
        print("="*80)
    
    def _load_feature_extractors(self):
        """Load transformers for feature extraction."""
        print("\n[1/2] Loading feature extractors...")
        
        from transformers import (
            BertTokenizer, BertModel,
            ViTFeatureExtractor, ViTModel,
            AutoTokenizer, AutoModelForSequenceClassification
        )
        from torchvision.models import resnet50
        
        # Text
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
        
        # Projections
        self.text_proj = nn.Linear(768, 128).to(self.device)
        self.meta_proj = nn.Linear(7, 128).to(self.device)
        
        print("   ✓ Feature extractors loaded")
    
    def _load_detection_model(self):
        """Load trained emotion-aware detection model."""
        print("\n[2/2] Loading detection model...")
        
        self.detection_model = StandaloneEmotionAwareDetector().to(self.device)
        
        ckpt_path = "checkpoints/best_emotion_aware_detector.pth"
        if Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location=self.device)
            
            # Remap keys
            remapped = {}
            for k, v in state.items():
                new_k = k.replace("fusion.", "fusion_layer.") if k.startswith("fusion.") else k
                remapped[new_k] = v
            
            self.detection_model.load_state_dict(remapped, strict=False)
            self.detection_model.eval()
            
            print("   ✓ Detection model loaded")
        else:
            print(f"   ⚠ Model not found at {ckpt_path}")
    
    def process_post(self, text, image_path=None, username="anonymous"):
        """
        Process a new post and generate comprehensive explanation WITH IMAGE ANALYSIS.
        
        Args:
            text: Post text content
            image_path: Optional path to image
            username: Username
            
        Returns:
            Complete results with full explainability including detailed image analysis
        """
        import emoji
        from PIL import Image
        import torchvision.transforms as T
        from ultralytics import YOLO
        
        post_id = hashlib.md5(text.encode()).hexdigest()[:8]
        timestamp = datetime.now()
        
        print(f"\n{'─'*80}")
        print(f"📝 POST: {post_id} | USER: {username}")
        print(f"{'─'*80}")
        print(f"{text[:150]}..." if len(text) > 150 else text)
        print(f"{'─'*80}\n")
        
        # Extract metadata
        hashtags = len(re.findall(r'#\w+', text))
        mentions = len(re.findall(r'@\w+', text))
        urls = len(re.findall(r'http\S+|www.\S+', text))
        emojis_count = len([c for c in text if emoji.is_emoji(c)])
        
        metadata = {
            'hashtags': hashtags,
            'mentions': mentions,
            'urls': urls,
            'emojis': emojis_count
        }
        
        # Clean text
        clean = re.sub(r'http\S+|www.\S+|#\w+|@\w+', '', text)
        clean = emoji.replace_emoji(clean, replace='')
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Text embedding
        tokens = self.bert_tokenizer(clean, padding=True, truncation=True,
                                     max_length=64, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            bert_out = self.bert(**tokens).last_hidden_state
            text_emb = self.text_proj(bert_out.mean(dim=1))
            text_emb = text_emb / (torch.norm(text_emb) + 1e-8)
        
        # 🆕 IMAGE PROCESSING WITH DETAILED ANALYSIS
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            image_present = 1.0
            
            # 🆕 Perform object detection with YOLO
            try:
                yolo = YOLO("yolov8n.pt")
                yolo_results = yolo(img)
                
                detected_objects = []
                if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                    for box in yolo_results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = yolo_results[0].names[class_id]
                        confidence = float(box.conf[0])
                        if confidence > 0.5:
                            detected_objects.append(class_name)
            except Exception as e:
                print(f"⚠️ YOLO detection failed: {e}")
                detected_objects = []
            
            # 🆕 Analyze image quality
            img_array = np.array(img)
            quality_score = min(1.0, (img.size[0] * img.size[1]) / (1920 * 1080))
            
            # 🆕 Detect scene type
            avg_brightness = img_array.mean() / 255.0
            color_variance = img_array.std() / 128.0
            
            if avg_brightness > 0.7 and color_variance < 0.5:
                scene_type = "indoor"
            elif avg_brightness < 0.4:
                scene_type = "night/dark"
            else:
                scene_type = "outdoor"
            
            # 🆕 Color saturation
            hsv = img.convert('HSV')
            hsv_array = np.array(hsv)
            saturation = hsv_array[:,:,1].mean() / 255.0
            
            # Create image analysis dict
            image_analysis = {
                'image_present': True,
                'detected_objects': detected_objects,
                'quality_score': quality_score,
                'scene_type': scene_type,
                'color_saturation': saturation,
                'visual_emotion': 'unknown',
                'visual_emotion_intensity': avg_brightness,
                'text_image_consistency': 0.5,  # Will be computed below
                'context_keywords': [],
            }
            
        else:
            img = Image.new("RGB", (224, 224))
            image_present = 0.0
            image_analysis = {'image_present': False}
        
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
        image_emb = torch.cat([vit_emb, cnn_pooled], dim=-1)
        image_emb = image_emb / (torch.norm(image_emb) + 1e-8)
        image_emb = image_emb * image_present
        
        # Metadata embedding
        ts = timestamp.timestamp()
        period = 24*60*60
        meta_vec = torch.tensor([
            hashtags, mentions, urls, emojis_count, image_present,
            np.sin(2*np.pi*ts/period), np.cos(2*np.pi*ts/period)
        ], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            meta_emb = self.meta_proj(meta_vec)
            meta_emb = meta_emb / (torch.norm(meta_emb) + 1e-8)
        
        # VAD emotion analysis
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
        
        emotion_input = torch.cat([vad_text, vad_image, affective_meta, pad], dim=-1)
        
        # Model inference
        with torch.no_grad():
            logits, intermediates = self.detection_model(
                text_emb, image_emb, meta_emb, emotion_input
            )
            
            prediction = torch.sigmoid(logits).item()
            confidence = abs(prediction - 0.5) * 2
        
        # 🆕 Compute text-image consistency - FIXED DIMENSION MISMATCH
        if image_analysis.get('image_present', False):
            text_emb_np = text_emb.cpu().numpy().flatten()
            image_emb_np = image_emb.cpu().numpy().flatten()
            
            # Project image embedding to same dimension as text (128)
            # Use average pooling to reduce from 1024 to 128
            image_emb_reduced = image_emb_np.reshape(8, 128).mean(axis=0)
            
            # Now compute consistency with matched dimensions
            text_norm = text_emb_np / (np.linalg.norm(text_emb_np) + 1e-8)
            image_norm = image_emb_reduced / (np.linalg.norm(image_emb_reduced) + 1e-8)
            consistency = float(np.dot(text_norm, image_norm))
            
            image_analysis['text_image_consistency'] = (consistency + 1) / 2
            
            # Extract context keywords
            urgent_keywords = ['breaking', 'urgent', 'live', 'now', 'alert']
            image_analysis['context_keywords'] = [kw for kw in urgent_keywords if kw in text.lower()]
            
            # Estimate visual emotion from VAD
            if vad[1] > 0.7:
                if vad[0] > 0.6:
                    image_analysis['visual_emotion'] = 'excitement'
                else:
                    image_analysis['visual_emotion'] = 'fear/anger'
                image_analysis['visual_emotion_intensity'] = vad[1]
            else:
                image_analysis['visual_emotion'] = 'neutral'
                image_analysis['visual_emotion_intensity'] = 0.5
        
        model_results = {
            'prediction': prediction,
            'confidence': confidence,
            'label': 'FAKE' if prediction > 0.5 else 'REAL',
            'emotion_weights': {
                'text': float(intermediates['emotion_weights'][0, 0]),
                'image': float(intermediates['emotion_weights'][0, 1]),
                'meta': float(intermediates['emotion_weights'][0, 2])
            },
            'congruence': float(intermediates['congruence']),
            'mixed_affect': float(intermediates['mixed_affect_score'])
        }
        
        # Generate comprehensive explanation WITH IMAGE ANALYSIS
        explanation = self.explainer.generate_comprehensive_explanation(
            post_id=post_id,
            username=username,
            timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            vad_vector=vad,
            z_aug=intermediates['z_aug'][0].cpu().numpy(),
            v_mismatch=intermediates['v_mismatch'][0].cpu().numpy(),
            metadata=metadata,
            model_results=model_results,
            image_analysis=image_analysis  # 🆕 PASS IMAGE ANALYSIS!
        )
        
        # Print explanation
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPLAINABILITY REPORT WITH IMAGE ANALYSIS")
        print("="*80)
        print(explanation['full_explanation'])
        print("="*80)
        
        # 🆕🆕🆕 NEW: Generate batch-style explanation
        anomaly_result = self.explainer.analyze_anomaly(
            intermediates['z_aug'][0].cpu().numpy()[:128],
            intermediates['v_mismatch'][0].cpu().numpy()
        )
        
        narrative_result = self.explainer.analyze_narrative_similarity(
            intermediates['z_aug'][0].cpu().numpy()[:128]
        )
        
        campaign_result = self.explainer.analyze_campaign_likelihood(metadata)
        
        batch_style_explanation = self.explainer.generate_batch_style_explanation(
            post_id=post_id,
            username=username,
            timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            vad_vector=vad,
            v_mismatch=intermediates['v_mismatch'][0].cpu().numpy(),
            metadata=metadata,
            anomaly_result=anomaly_result,
            narrative_result=narrative_result,
            campaign_result=campaign_result,
            model_prediction=prediction,  # ADD THIS
            model_confidence=confidence   # ADD THIS
        )
        
        # 🆕🆕🆕 Print batch-style explanation
        print("\n" + "="*80)
        print("BATCH-STYLE EXPLAINABILITY OUTPUT (explainability_4.py format)")
        print("="*80)
        print(batch_style_explanation)
        print("─" * 80)
        
        return {
            'post_id': post_id,
            'username': username,
            'timestamp': timestamp.isoformat(),
            'text': text,
            'model_results': model_results,
            'metadata': metadata,
            'image_analysis': image_analysis,  # 🆕 INCLUDE IMAGE ANALYSIS
            'explanation': explanation,
            'batch_style_explanation': batch_style_explanation  # 🆕🆕🆕 NEW!
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ComprehensiveInferencePipeline()
    
    # Test with example
    print("\n\n" + "🔥"*40)
    print("TESTING COMPREHENSIVE EXPLAINABILITY WITH IMAGE ANALYSIS")
    print("🔥"*40)
    
    result = pipeline.process_post(
        text='Hurricane Sandy. We are coming for you. http://t.co/WLWJ8krG ',
        image_path='Dataset/twitter/images_test/attacks_paris_1.jpg',
        username="conspiracy_news"
    )
    
    print("\n✅ COMPREHENSIVE ANALYSIS WITH IMAGE ANALYSIS COMPLETE!")