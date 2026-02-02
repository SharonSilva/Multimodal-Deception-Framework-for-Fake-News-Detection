"""
Model Interpretation Tool - What Does the Emotion-Aware Model Learn?
====================================================================
Analyze and visualize what patterns the model captures
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# 1. EXTRACT LEARNED PATTERNS
# ============================================================================
def analyze_model_internals(model, dataloader, device, max_batches=10):
    """
    Extract intermediate representations to understand what the model learns.
    
    Returns:
        Dict containing:
        - attention_weights: Cross-modal attention patterns
        - emotion_gates: How emotions modulate fusion
        - modality_weights: Which modalities are trusted
        - mismatch_vectors: Text-image contradiction signals
        - congruence_scores: Emotional consistency across modalities
    """
    model.eval()
    
    collected = {
        'attention_weights': [],
        'emotion_gates_text': [],
        'emotion_gates_image': [],
        'modality_weights': [],
        'mismatch_vectors': [],
        'congruence_scores': [],
        'vad_text': [],
        'vad_image': [],
        'predictions': [],
        'labels': [],
        'z_fused': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting learned patterns")):
            if i >= max_batches:
                break
                
            h_text = batch['text_features'].to(device)
            h_image = batch['image_features'].to(device)
            h_meta = batch['metadata_features'].to(device)
            vad_text = batch['vad_text'].to(device)
            vad_image = batch['vad_image'].to(device)
            affective_meta = batch['affective_meta'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with intermediates
            logits, intermediates = model(h_text, h_image, h_meta, affective_meta, vad_text, vad_image)
            preds = torch.sigmoid(logits.squeeze())
            
            # Collect data
            if 'attn_weights' in intermediates:
                collected['attention_weights'].append(intermediates['attn_weights'].cpu())
            if 'emotion_gate_text' in intermediates:
                collected['emotion_gates_text'].append(intermediates['emotion_gate_text'].cpu())
            if 'emotion_gate_image' in intermediates:
                collected['emotion_gates_image'].append(intermediates['emotion_gate_image'].cpu())
            if 'modality_weights' in intermediates:
                collected['modality_weights'].append(intermediates['modality_weights'].cpu())
            if 'v_mismatch' in intermediates:
                mismatch = intermediates['v_mismatch']
                if mismatch.ndim > 2:
                    mismatch = mismatch.squeeze(1)
                collected['mismatch_vectors'].append(mismatch.cpu())
            if 'congruence' in intermediates:
                collected['congruence_scores'].append(intermediates['congruence'].cpu())
            if 'z_fused' in intermediates:
                z_fused = intermediates['z_fused']
                if z_fused.ndim > 2:
                    z_fused = z_fused.squeeze(1)
                collected['z_fused'].append(z_fused.cpu())
            
            collected['vad_text'].append(vad_text.cpu())
            collected['vad_image'].append(vad_image.cpu())
            collected['predictions'].append(preds.cpu())
            collected['labels'].append(labels.cpu())
    
    # Concatenate all batches
    for key in collected:
        if collected[key]:
            collected[key] = torch.cat(collected[key], dim=0).numpy()
    
    return collected

# ============================================================================
# 2. VISUALIZE ATTENTION PATTERNS
# ============================================================================
def visualize_attention_patterns(collected_data, save_path='attention_analysis.png'):
    """Visualize what cross-modal attention patterns the model learned."""
    
    if 'attention_weights' not in collected_data or len(collected_data['attention_weights']) == 0:
        print("⚠️  No attention weights found in model")
        return
    
    attn = collected_data['attention_weights']
    labels = collected_data['labels']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('What Cross-Modal Attention Learns', fontsize=16, fontweight='bold')
    
    # Average attention for fake vs real
    fake_mask = labels == 0
    real_mask = labels == 1
    
    if fake_mask.any():
        fake_attn_avg = attn[fake_mask].mean(axis=0)
        axes[0, 0].imshow(fake_attn_avg, cmap='RdYlBu_r', aspect='auto')
        axes[0, 0].set_title('Average Attention: FAKE News')
        axes[0, 0].set_xlabel('Image Tokens')
        axes[0, 0].set_ylabel('Text Tokens')
        plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
    
    if real_mask.any():
        real_attn_avg = attn[real_mask].mean(axis=0)
        axes[0, 1].imshow(real_attn_avg, cmap='RdYlBu_r', aspect='auto')
        axes[0, 1].set_title('Average Attention: REAL News')
        axes[0, 1].set_xlabel('Image Tokens')
        axes[0, 1].set_ylabel('Text Tokens')
        plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    
    # Attention distribution
    axes[1, 0].hist(attn[fake_mask].flatten(), bins=50, alpha=0.6, label='Fake', color='red')
    axes[1, 0].hist(attn[real_mask].flatten(), bins=50, alpha=0.6, label='Real', color='blue')
    axes[1, 0].set_title('Attention Weight Distribution')
    axes[1, 0].set_xlabel('Attention Weight')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Attention sparsity (how focused is attention?)
    fake_entropy = -np.sum(attn[fake_mask] * np.log(attn[fake_mask] + 1e-8), axis=(1, 2))
    real_entropy = -np.sum(attn[real_mask] * np.log(attn[real_mask] + 1e-8), axis=(1, 2))
    
    axes[1, 1].boxplot([fake_entropy, real_entropy], labels=['Fake', 'Real'])
    axes[1, 1].set_title('Attention Focus (Lower = More Focused)')
    axes[1, 1].set_ylabel('Entropy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved attention analysis to {save_path}")
    plt.show()

# ============================================================================
# 3. VISUALIZE EMOTION GATING
# ============================================================================
def visualize_emotion_gating(collected_data, save_path='emotion_gates.png'):
    """Show how emotions modulate the fusion process."""
    
    if 'emotion_gates_text' not in collected_data or len(collected_data['emotion_gates_text']) == 0:
        print("⚠️  No emotion gates found in model")
        return
    
    gate_text = collected_data['emotion_gates_text']
    gate_image = collected_data['emotion_gates_image']
    vad_text = collected_data['vad_text']
    vad_image = collected_data['vad_image']
    labels = collected_data['labels']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Emotion Gating Mechanisms', fontsize=16, fontweight='bold')
    
    fake_mask = labels == 0
    real_mask = labels == 1
    
    # Text emotion gates
    for i, emotion in enumerate(['Valence', 'Arousal', 'Dominance']):
        axes[0, i].scatter(vad_text[fake_mask, i], gate_text[fake_mask], 
                          alpha=0.3, c='red', label='Fake', s=10)
        axes[0, i].scatter(vad_text[real_mask, i], gate_text[real_mask], 
                          alpha=0.3, c='blue', label='Real', s=10)
        axes[0, i].set_xlabel(f'Text {emotion}')
        axes[0, i].set_ylabel('Gate Activation')
        axes[0, i].set_title(f'Text Gate vs {emotion}')
        axes[0, i].legend()
    
    # Image emotion gates
    for i, emotion in enumerate(['Valence', 'Arousal', 'Dominance']):
        axes[1, i].scatter(vad_image[fake_mask, i], gate_image[fake_mask], 
                          alpha=0.3, c='red', label='Fake', s=10)
        axes[1, i].scatter(vad_image[real_mask, i], gate_image[real_mask], 
                          alpha=0.3, c='blue', label='Real', s=10)
        axes[1, i].set_xlabel(f'Image {emotion}')
        axes[1, i].set_ylabel('Gate Activation')
        axes[1, i].set_title(f'Image Gate vs {emotion}')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved emotion gating analysis to {save_path}")
    plt.show()

# ============================================================================
# 4. VISUALIZE MISMATCH VECTORS
# ============================================================================
def visualize_mismatch_patterns(collected_data, save_path='mismatch_analysis.png'):
    """Analyze text-image contradiction patterns."""
    
    if 'mismatch_vectors' not in collected_data or len(collected_data['mismatch_vectors']) == 0:
        print("⚠️  No mismatch vectors found")
        return
    
    mismatch = collected_data['mismatch_vectors']
    labels = collected_data['labels']
    congruence = collected_data.get('congruence_scores', None)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Text-Image Mismatch Patterns', fontsize=16, fontweight='bold')
    
    fake_mask = labels == 0
    real_mask = labels == 1
    
    # Mismatch magnitude
    mismatch_mag = np.linalg.norm(mismatch, axis=1)
    
    axes[0, 0].hist(mismatch_mag[fake_mask], bins=50, alpha=0.6, color='red', label='Fake')
    axes[0, 0].hist(mismatch_mag[real_mask], bins=50, alpha=0.6, color='blue', label='Real')
    axes[0, 0].set_title('Mismatch Magnitude Distribution')
    axes[0, 0].set_xlabel('L2 Norm of Mismatch Vector')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].axvline(mismatch_mag[fake_mask].mean(), color='red', linestyle='--', 
                       label=f'Fake Mean: {mismatch_mag[fake_mask].mean():.3f}')
    axes[0, 0].axvline(mismatch_mag[real_mask].mean(), color='blue', linestyle='--',
                       label=f'Real Mean: {mismatch_mag[real_mask].mean():.3f}')
    
    # Mismatch PCA
    pca = PCA(n_components=2)
    mismatch_2d = pca.fit_transform(mismatch)
    
    axes[0, 1].scatter(mismatch_2d[fake_mask, 0], mismatch_2d[fake_mask, 1], 
                      c='red', alpha=0.3, label='Fake', s=10)
    axes[0, 1].scatter(mismatch_2d[real_mask, 0], mismatch_2d[real_mask, 1], 
                      c='blue', alpha=0.3, label='Real', s=10)
    axes[0, 1].set_title(f'Mismatch Space (PCA, variance: {pca.explained_variance_ratio_.sum():.2%})')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].legend()
    
    # Congruence scores
    if congruence is not None:
        axes[1, 0].boxplot([congruence[fake_mask], congruence[real_mask]], 
                          labels=['Fake', 'Real'])
        axes[1, 0].set_title('Emotional Congruence Scores')
        axes[1, 0].set_ylabel('Congruence (0=mismatch, 1=aligned)')
        
        # Congruence vs Prediction confidence
        predictions = collected_data['predictions']
        axes[1, 1].scatter(congruence[fake_mask], predictions[fake_mask], 
                          c='red', alpha=0.3, label='Fake', s=10)
        axes[1, 1].scatter(congruence[real_mask], predictions[real_mask], 
                          c='blue', alpha=0.3, label='Real', s=10)
        axes[1, 1].set_xlabel('Emotional Congruence')
        axes[1, 1].set_ylabel('Prediction Confidence')
        axes[1, 1].set_title('Congruence vs Model Confidence')
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No congruence data', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'No congruence data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved mismatch analysis to {save_path}")
    plt.show()

# ============================================================================
# 5. VISUALIZE MODALITY IMPORTANCE
# ============================================================================
def visualize_modality_weights(collected_data, save_path='modality_weights.png'):
    """Show which modalities the model trusts for fake vs real news."""
    
    if 'modality_weights' not in collected_data or len(collected_data['modality_weights']) == 0:
        print("⚠️  No modality weights found")
        return
    
    weights = collected_data['modality_weights']
    labels = collected_data['labels']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Adaptive Modality Weighting', fontsize=16, fontweight='bold')
    
    fake_mask = labels == 0
    real_mask = labels == 1
    
    modalities = ['Text', 'Image', 'Metadata']
    
    # Average weights
    fake_weights = weights[fake_mask].mean(axis=0)
    real_weights = weights[real_mask].mean(axis=0)
    
    x = np.arange(len(modalities))
    width = 0.35
    
    axes[0].bar(x - width/2, fake_weights, width, label='Fake', color='red', alpha=0.7)
    axes[0].bar(x + width/2, real_weights, width, label='Real', color='blue', alpha=0.7)
    axes[0].set_ylabel('Average Weight')
    axes[0].set_title('Average Modality Weights')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modalities)
    axes[0].legend()
    
    # Weight distributions
    for i, modality in enumerate(modalities):
        axes[1].hist(weights[fake_mask, i], bins=30, alpha=0.4, color='red', 
                    label=f'Fake {modality}')
        axes[1].hist(weights[real_mask, i], bins=30, alpha=0.4, color='blue', 
                    label=f'Real {modality}')
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Weight Distributions')
    axes[1].legend()
    
    # Weight correlation
    corr_fake = np.corrcoef(weights[fake_mask].T)
    corr_real = np.corrcoef(weights[real_mask].T)
    
    im = axes[2].imshow(corr_fake - corr_real, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(modalities)))
    axes[2].set_yticks(range(len(modalities)))
    axes[2].set_xticklabels(modalities)
    axes[2].set_yticklabels(modalities)
    axes[2].set_title('Weight Correlation Diff (Fake - Real)')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved modality weights to {save_path}")
    plt.show()

# ============================================================================
# 6. VISUALIZE EMBEDDING SPACE
# ============================================================================
def visualize_embedding_space(collected_data, save_path='embedding_space.png'):
    """Visualize how the model separates fake vs real news in the fused space."""
    
    if 'z_fused' not in collected_data or len(collected_data['z_fused']) == 0:
        print("⚠️  No fused embeddings found")
        return
    
    z_fused = collected_data['z_fused']
    labels = collected_data['labels']
    predictions = collected_data['predictions']
    
    # Use t-SNE for visualization
    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z_fused)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Learned Embedding Space', fontsize=16, fontweight='bold')
    
    # Color by true labels
    scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=labels, 
                              cmap='RdBu_r', alpha=0.6, s=20)
    axes[0].set_title('Colored by True Labels')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0], label='0=Fake, 1=Real')
    
    # Color by prediction confidence
    scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=predictions, 
                              cmap='RdYlGn', alpha=0.6, s=20)
    axes[1].set_title('Colored by Prediction Confidence')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1], label='Confidence')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved embedding space to {save_path}")
    plt.show()

# ============================================================================
# 7. MAIN ANALYSIS FUNCTION
# ============================================================================
def analyze_what_model_learns(model, dataloader, device, output_dir='model_analysis'):
    """
    Complete analysis of what the emotion-aware model learns.
    
    Args:
        model: Trained EmotionAwareFakeNewsDetector
        dataloader: DataLoader with validation/test data
        device: torch device
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🔬 ANALYZING WHAT THE MODEL LEARNS")
    print("="*70 + "\n")
    
    # Extract learned patterns
    collected = analyze_model_internals(model, dataloader, device, max_batches=50)
    
    print("\n📊 Generating visualizations...\n")
    
    # Generate all visualizations
    visualize_attention_patterns(collected, f'{output_dir}/attention_analysis.png')
    visualize_emotion_gating(collected, f'{output_dir}/emotion_gates.png')
    visualize_mismatch_patterns(collected, f'{output_dir}/mismatch_analysis.png')
    visualize_modality_weights(collected, f'{output_dir}/modality_weights.png')
    visualize_embedding_space(collected, f'{output_dir}/embedding_space.png')
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📈 SUMMARY STATISTICS")
    print("="*70)
    
    fake_mask = collected['labels'] == 0
    real_mask = collected['labels'] == 1
    
    if 'mismatch_vectors' in collected and len(collected['mismatch_vectors']) > 0:
        mismatch_mag = np.linalg.norm(collected['mismatch_vectors'], axis=1)
        print(f"\n🔹 Mismatch Magnitude:")
        print(f"   Fake News: {mismatch_mag[fake_mask].mean():.4f} ± {mismatch_mag[fake_mask].std():.4f}")
        print(f"   Real News: {mismatch_mag[real_mask].mean():.4f} ± {mismatch_mag[real_mask].std():.4f}")
    
    if 'congruence_scores' in collected and len(collected['congruence_scores']) > 0:
        congruence = collected['congruence_scores']
        print(f"\n🔹 Emotional Congruence:")
        print(f"   Fake News: {congruence[fake_mask].mean():.4f} ± {congruence[fake_mask].std():.4f}")
        print(f"   Real News: {congruence[real_mask].mean():.4f} ± {congruence[real_mask].std():.4f}")
    
    if 'modality_weights' in collected and len(collected['modality_weights']) > 0:
        weights = collected['modality_weights']
        print(f"\n🔹 Modality Weights (Text, Image, Metadata):")
        print(f"   Fake News: [{weights[fake_mask, 0].mean():.3f}, {weights[fake_mask, 1].mean():.3f}, {weights[fake_mask, 2].mean():.3f}]")
        print(f"   Real News: [{weights[real_mask, 0].mean():.3f}, {weights[real_mask, 1].mean():.3f}, {weights[real_mask, 2].mean():.3f}]")
    
    print("\n" + "="*70)
    print(f"✅ Analysis complete! Results saved to '{output_dir}/'")
    print("="*70 + "\n")
    
    return collected

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Load your trained model
    from train_emotion_gated import EmotionAwareFakeNewsDetector, FakeNewsVADDataset
    
    model = EmotionAwareFakeNewsDetector().to(device)
    model.load_state_dict(torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=device))
    model.eval()
    
    # Load validation data (you'll need to recreate this from your training script)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Run analysis
    # collected_data = analyze_what_model_learns(model, val_loader, device)
    
    print("📖 To use this tool:")
    print("   1. Import: from model_interpretation import analyze_what_model_learns")
    print("   2. Load your model and data")
    print("   3. Run: analyze_what_model_learns(model, val_loader, device)")