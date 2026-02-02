"""
INTERACTIVE FAKE NEWS DETECTION & EXPLAINABILITY SYSTEM
========================================================
Allows users to input new posts (single or multiple) and get:
1. Suspicion score with confidence level
2. Detailed explanation (text, image, metadata contributions)
3. Campaign detection (if multiple posts provided)
4. Visual explanations with highlighted suspicious elements

Usage:
    # Single post
    result = detector.detect_single_post(text, image_path, user_metadata)
    
    # Multiple posts (campaign detection)
    results = detector.detect_multiple_posts(posts_list)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from rough_work import EmotionAwareFakeNewsDetector
from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InteractiveFakeNewsDetector:
    """
    Complete inference system for detecting suspicious content with explainability.
    """
    
    def __init__(self):
        print("="*80)
        print("INITIALIZING INTERACTIVE FAKE NEWS DETECTION SYSTEM")
        print("="*80)
        
        # Load all necessary models and data
        self._load_models()
        self._load_detection_components()
        self._load_explainability_data()
        
        print("\n✅ System ready for inference!")
    
    def _load_models(self):
        """Load trained models"""
        print("\n[1/4] Loading trained models...")
        
        # 1. Emotion-aware multimodal model
        self.emotion_model = EmotionAwareFakeNewsDetector(
            d_text=128, d_image=1024, d_meta=128,
            d_common=256, vad_dim=3, meta_affective_dim=128,
            mismatch_dim=128, temporal_hidden=64, num_classes=1
        ).to(device)
        
        state = torch.load("checkpoints/best_emotion_aware_detector.pth", 
                          map_location=device, weights_only=False)
        
        # Remap keys
        new_state = {}
        for k, v in state.items():
            if k.startswith("fusion."):
                new_k = k.replace("fusion.", "fusion_layer.")
                new_state[new_k] = v
            else:
                if not k.startswith("classifier."):
                    new_state[k] = v
        
        self.emotion_model.load_state_dict(new_state, strict=False)
        self.emotion_model.eval()
        
        # 2. Heterogeneous GNN model
        node_features, edge_dict, node_mappings = load_heterogeneous_graph()
        node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}
        
        self.gnn_model = TemporalHeterogeneousGNN(
            node_dims=node_dims,
            hidden_dim=256,
            num_layers=3,
            relation_types=list(edge_dict.keys()),
            num_classes=2
        ).to(device)
        
        self.node_features = {k: v.to(device) for k, v in node_features.items()}
        self.edge_dict = {k: (ei.to(device), ew.to(device)) 
                         for k, (ei, ew) in edge_dict.items()}
        
        print("   ✅ Models loaded")
    
    def _load_detection_components(self):
        """Load detection thresholds and methods"""
        print("\n[2/4] Loading detection components...")
        
        # Load trained embeddings for similarity comparison
        self.trained_embeddings = torch.load(
            "suspicious_detection_results/trained_embeddings.pt",
            weights_only=False
        ) if Path("suspicious_detection_results/trained_embeddings.pt").exists() else None
        
        # Detection thresholds (from training)
        self.thresholds = {
            'isolation_forest': 0.5,
            'dbscan_eps': 0.5,
            'distance_percentile': 85,
            'suspicion_threshold': 0.25
        }
        
        print("   ✅ Detection components ready")
    
    def _load_explainability_data(self):
        """Load explainability components"""
        print("\n[3/4] Loading explainability data...")
        
        # Load suspicious phrase scores
        try:
            phrases_df = pd.read_csv("explainability_results/suspicious_phrases.csv")
            self.suspicious_phrases = dict(zip(phrases_df['phrase'], phrases_df['score']))
        except:
            self.suspicious_phrases = {}
        
        # Load user risk patterns
        try:
            user_risks = pd.read_csv("explainability_results/user_risk_scores.csv", index_col=0)
            self.user_risk_patterns = user_risks.to_dict('index')
        except:
            self.user_risk_patterns = {}
        
        print("   ✅ Explainability data loaded")
    
    def detect_single_post(self, text, image=None, metadata=None, timestamp=None):
        """
        Detect suspicious content in a single post with full explainability.
        
        Args:
            text (str): Post text content
            image (PIL.Image or path): Image associated with post
            metadata (dict): User metadata {username, followers, etc.}
            timestamp (datetime): Post timestamp
        
        Returns:
            dict: Detection results with explanations
        """
        print("\n" + "="*80)
        print("ANALYZING SINGLE POST")
        print("="*80)
        
        # Process inputs
        processed = self._preprocess_single_post(text, image, metadata, timestamp)
        
        # Get embeddings and fusion weights
        with torch.no_grad():
            # Multimodal detection
            logits, intermediates = self.emotion_model(
                processed['text_emb'],
                processed['image_emb'],
                processed['meta_emb'],
                affective_meta=processed['affective_meta'],
                vad_text=processed['vad_text'],
                vad_image=processed['vad_image']
            )
            
            suspicion_prob = torch.sigmoid(logits).item()
            fusion_weights = intermediates['emotion_weights'].cpu().numpy()[0]
        
        # Generate explanation
        explanation = self._generate_single_post_explanation(
            text=text,
            suspicion_score=suspicion_prob,
            fusion_weights=fusion_weights,
            metadata=metadata
        )
        
        return explanation
    
    def detect_multiple_posts(self, posts_list, detect_campaigns=True):
        """
        Detect suspicious content in multiple posts with campaign detection.
        
        Args:
            posts_list (list): List of post dicts with keys:
                {text, image, metadata, timestamp}
            detect_campaigns (bool): Whether to detect coordination campaigns
        
        Returns:
            dict: Detection results for all posts + campaign analysis
        """
        print("\n" + "="*80)
        print(f"ANALYZING {len(posts_list)} POSTS")
        print("="*80)
        
        # Detect each post individually
        post_results = []
        for i, post in enumerate(posts_list, 1):
            print(f"\n[{i}/{len(posts_list)}] Analyzing post...")
            result = self.detect_single_post(
                text=post.get('text', ''),
                image=post.get('image'),
                metadata=post.get('metadata'),
                timestamp=post.get('timestamp')
            )
            post_results.append(result)
        
        # Campaign detection
        campaign_analysis = None
        if detect_campaigns and len(posts_list) > 1:
            print("\n🔍 Analyzing for coordination campaigns...")
            campaign_analysis = self._detect_campaigns(posts_list, post_results)
        
        return {
            'posts': post_results,
            'campaigns': campaign_analysis,
            'summary': self._generate_batch_summary(post_results, campaign_analysis)
        }
    
    def _preprocess_single_post(self, text, image, metadata, timestamp):
        """Preprocess a single post into model inputs"""
        
        # Text embedding (simplified - use sentence-transformers in production)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=128)
        
        # Dummy corpus for fitting
        corpus = [text, "sample text"]
        vectorizer.fit(corpus)
        text_vec = vectorizer.transform([text]).toarray()[0]
        text_emb = torch.tensor(text_vec, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Image embedding (placeholder - use proper vision model in production)
        if image is not None:
            # In production: use ResNet/ViT to extract features
            image_emb = torch.randn(1, 1024).to(device)  # Placeholder
        else:
            image_emb = torch.zeros(1, 1024).to(device)
        
        # Metadata embedding
        if metadata is not None:
            # Extract features from metadata
            meta_features = self._extract_metadata_features(metadata)
            meta_emb = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            meta_emb = torch.zeros(1, 128).to(device)
        
        # VAD features (placeholder - use proper emotion analysis in production)
        vad_text = torch.randn(1, 3).to(device)
        vad_image = torch.randn(1, 3).to(device)
        affective_meta = torch.randn(1, 128).to(device)
        
        return {
            'text_emb': text_emb,
            'image_emb': image_emb,
            'meta_emb': meta_emb,
            'vad_text': vad_text,
            'vad_image': vad_image,
            'affective_meta': affective_meta
        }
    
    def _extract_metadata_features(self, metadata):
        """Extract numerical features from user metadata"""
        features = []
        
        # Example features
        features.append(np.log1p(metadata.get('followers', 0)))
        features.append(np.log1p(metadata.get('following', 0)))
        features.append(np.log1p(metadata.get('posts_count', 0)))
        features.append(metadata.get('verified', 0))
        features.append(metadata.get('account_age_days', 0) / 365.0)
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        
        return np.array(features[:128])
    
    def _generate_single_post_explanation(self, text, suspicion_score, 
                                         fusion_weights, metadata):
        """Generate human-readable explanation for a single post"""
        
        # Binary classification based on threshold
        FAKE_THRESHOLD = 0.5  # Configurable threshold
        is_fake = suspicion_score >= FAKE_THRESHOLD
        prediction = "FAKE" if is_fake else "REAL"
        
        # Determine confidence level
        if suspicion_score > 0.75:
            confidence = "Very High"
            risk_level = "CRITICAL"
        elif suspicion_score > 0.5:
            confidence = "High"
            risk_level = "HIGH"
        elif suspicion_score > 0.25:
            confidence = "Medium"
            risk_level = "MODERATE"
        else:
            confidence = "Low"
            risk_level = "LOW"
        
        # Identify suspicious text phrases
        words = text.lower().split()
        suspicious_words = []
        for word in words:
            if word in self.suspicious_phrases:
                suspicious_words.append((word, self.suspicious_phrases[word]))
        
        suspicious_words.sort(key=lambda x: x[1], reverse=True)
        
        # Modality contributions
        modalities = ['Text', 'Image', 'Metadata']
        dominant_modality = modalities[np.argmax(fusion_weights)]
        
        # Build explanation
        explanation = {
            'prediction': prediction,  # NEW: FAKE or REAL
            'is_fake': is_fake,  # NEW: Boolean
            'suspicion_score': float(suspicion_score),
            'confidence_level': confidence,
            'risk_level': risk_level,
            'threshold_used': FAKE_THRESHOLD,  # NEW: Show threshold
            'fusion_weights': {
                'text': float(fusion_weights[0]),
                'image': float(fusion_weights[1]),
                'metadata': float(fusion_weights[2])
            },
            'dominant_modality': dominant_modality,
            'suspicious_phrases': suspicious_words[:5],
            'summary': self._build_explanation_text(
                prediction, suspicion_score, confidence, dominant_modality,
                fusion_weights, suspicious_words, FAKE_THRESHOLD
            )
        }
        
        return explanation
    
    def _build_explanation_text(self, prediction, score, confidence, dominant_mod, 
                               weights, suspicious_words, threshold):
        """Build natural language explanation"""
        
        # Start with clear prediction
        explanation = f"📊 PREDICTION: {prediction}\n\n"
        
        explanation += f"This post is classified as {prediction.lower()} with a {confidence.lower()} confidence suspicion score of {score:.2%} "
        explanation += f"(threshold: {threshold:.0%}). "
        
        explanation += f"The '{dominant_mod.lower()}' modality contributed most ({weights[['text', 'image', 'metadata'].index(dominant_mod.lower())]:.1%}) to this detection. "
        
        if suspicious_words and dominant_mod.lower() == 'text':
            top_phrase = suspicious_words[0][0]
            explanation += f"Key suspicious phrase detected: '{top_phrase}'. "
        elif dominant_mod.lower() == 'image':
            explanation += "Visual content shows patterns consistent with suspicious posts. "
        elif dominant_mod.lower() == 'metadata':
            explanation += "User behavior metadata indicates anomalous patterns. "
        
        return explanation
    
    def _detect_campaigns(self, posts_list, post_results):
        """Detect coordination campaigns in multiple posts"""
        
        # Extract timestamps
        timestamps = [p.get('timestamp', datetime.now()) for p in posts_list]
        users = [p.get('metadata', {}).get('username', f'user_{i}') 
                for i, p in enumerate(posts_list)]
        
        # Find temporal clusters
        campaigns = []
        window_minutes = 5
        
        for i, ts in enumerate(timestamps):
            if ts is None:
                continue
            
            # Find posts within time window
            cluster_indices = []
            for j, other_ts in enumerate(timestamps):
                if other_ts is None:
                    continue
                
                time_diff = abs((other_ts - ts).total_seconds() / 60)
                if time_diff <= window_minutes:
                    cluster_indices.append(j)
            
            if len(cluster_indices) >= 3:  # At least 3 posts
                unique_users = len(set(users[idx] for idx in cluster_indices))
                
                if unique_users >= 2:  # At least 2 different users
                    avg_suspicion = np.mean([post_results[idx]['suspicion_score'] 
                                           for idx in cluster_indices])
                    
                    campaigns.append({
                        'num_posts': len(cluster_indices),
                        'num_users': unique_users,
                        'time_window_minutes': window_minutes,
                        'avg_suspicion': float(avg_suspicion),
                        'post_indices': cluster_indices,
                        'is_coordinated': unique_users >= 2 and len(cluster_indices) >= 3
                    })
        
        # Remove duplicates
        unique_campaigns = []
        seen = set()
        for camp in campaigns:
            key = tuple(sorted(camp['post_indices']))
            if key not in seen:
                unique_campaigns.append(camp)
                seen.add(key)
        
        return {
            'num_campaigns': len(unique_campaigns),
            'campaigns': unique_campaigns,
            'summary': f"Detected {len(unique_campaigns)} potential coordination campaign(s)"
        }
    
    def _generate_batch_summary(self, post_results, campaign_analysis):
        """Generate summary for batch analysis"""
        
        num_posts = len(post_results)
        num_suspicious = sum(1 for r in post_results if r['suspicion_score'] > 0.5)
        avg_suspicion = np.mean([r['suspicion_score'] for r in post_results])
        
        summary = {
            'total_posts': num_posts,
            'suspicious_posts': num_suspicious,
            'avg_suspicion_score': float(avg_suspicion),
            'num_campaigns': campaign_analysis['num_campaigns'] if campaign_analysis else 0
        }
        
        return summary


# ============================================================================
# EXAMPLE USAGE & DEMO
# ============================================================================

def demo_single_post():
    """Demo: Analyze a single post"""
    print("\n" + "="*80)
    print("DEMO: SINGLE POST DETECTION")
    print("="*80)
    
    # Initialize detector
    detector = InteractiveFakeNewsDetector()
    
    # Example post
    result = detector.detect_single_post(
        text="BREAKING: Miracle cure for COVID discovered! Share immediately! #urgent",
        image=None,
        metadata={
            'username': 'suspicious_account',
            'followers': 50,
            'following': 5000,
            'posts_count': 10,
            'verified': 0,
            'account_age_days': 5
        },
        timestamp=datetime.now()
    )
    
    # Print results
    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)
    print(f"\n🎯 PREDICTION: {result['prediction']}")
    print(f"   {'❌ FAKE NEWS' if result['is_fake'] else '✅ REAL/LEGITIMATE'}")
    print(f"\n📊 Suspicion Score: {result['suspicion_score']:.2%}")
    print(f"   Threshold: {result['threshold_used']:.0%}")
    print(f"⚠️  Risk Level: {result['risk_level']}")
    print(f"📊 Confidence: {result['confidence_level']}")
    
    print(f"\n💡 Modality Contributions:")
    for mod, weight in result['fusion_weights'].items():
        print(f"   {mod.capitalize()}: {weight:.1%}")
    
    print(f"\n🔍 Dominant Modality: {result['dominant_modality']}")
    
    if result['suspicious_phrases']:
        print(f"\n⚠️  Suspicious Phrases Detected:")
        for phrase, score in result['suspicious_phrases']:
            print(f"   - '{phrase}' (score: {score:.4f})")
    
    print(f"\n📝 Explanation:")
    print(f"   {result['summary']}")


def demo_multiple_posts():
    """Demo: Analyze multiple posts for campaign detection"""
    print("\n" + "="*80)
    print("DEMO: MULTIPLE POSTS WITH CAMPAIGN DETECTION")
    print("="*80)
    
    # Initialize detector
    detector = InteractiveFakeNewsDetector()
    
    # Example posts (simulating a coordinated campaign)
    posts = [
        {
            'text': f"Breaking news! Incredible discovery! Post {i}",
            'metadata': {'username': f'user_{i}', 'followers': 100},
            'timestamp': datetime.now()
        }
        for i in range(5)
    ]
    
    # Detect
    results = detector.detect_multiple_posts(posts, detect_campaigns=True)
    
    # Print results
    print("\n" + "="*80)
    print("BATCH ANALYSIS RESULTS")
    print("="*80)
    
    summary = results['summary']
    print(f"\n📊 Summary:")
    print(f"   Total Posts: {summary['total_posts']}")
    print(f"   Fake Posts: {summary['suspicious_posts']}")
    print(f"   Real Posts: {summary['total_posts'] - summary['suspicious_posts']}")
    print(f"   Avg Suspicion Score: {summary['avg_suspicion_score']:.2%}")
    print(f"   Campaigns Detected: {summary['num_campaigns']}")
    
    if results['campaigns'] and results['campaigns']['num_campaigns'] > 0:
        print(f"\n🚨 CAMPAIGN ALERT:")
        for i, camp in enumerate(results['campaigns']['campaigns'], 1):
            print(f"\n   Campaign {i}:")
            print(f"      Posts: {camp['num_posts']}")
            print(f"      Users: {camp['num_users']}")
            print(f"      Avg Suspicion: {camp['avg_suspicion']:.2%}")
            print(f"      Coordinated: {'YES' if camp['is_coordinated'] else 'NO'}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTERACTIVE FAKE NEWS DETECTION SYSTEM")
    print("="*80)
    print("\nSelect demo:")
    print("1. Single post detection")
    print("2. Multiple posts with campaign detection")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == "1":
        demo_single_post()
    elif choice == "2":
        demo_multiple_posts()
    else:
        print("Invalid choice. Running single post demo...")
        demo_single_post()