"""
ROOT CAUSE ANALYSIS FOR POOR MODEL PERFORMANCE
===============================================

This script systematically checks each potential cause of failure.
Run this BEFORE retraining to identify the real problem.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# ROOT CAUSE 1: ARE THE LABELS CORRECT?
# ==============================================================================

def check_labels(df, label_col='label'):
    """Verify label quality"""
    print("\n" + "="*80)
    print("ROOT CAUSE 1: LABEL QUALITY CHECK")
    print("="*80)
    
    if label_col not in df.columns:
        print("❌ CRITICAL: No label column found in DataFrame!")
        print(f"   Available columns: {df.columns.tolist()}")
        print("\n⚠️  YOU NEED TO CREATE LABELS FIRST!")
        print("   Example: df['label'] = 0 for real, 1 for fake")
        return False
    
    labels = df[label_col].values
    unique_labels = np.unique(labels)
    
    print(f"Label distribution:")
    for label in unique_labels:
        count = sum(labels == label)
        print(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Check if labels are binary
    if len(unique_labels) != 2:
        print(f"❌ ERROR: Expected 2 classes, found {len(unique_labels)}")
        return False
    
    # Check for extreme imbalance
    counts = [sum(labels == l) for l in unique_labels]
    ratio = min(counts) / max(counts)
    if ratio < 0.1:
        print(f"⚠️  WARNING: Severe class imbalance (ratio={ratio:.3f})")
        print("   Consider using class weights or resampling")
    
    print("✅ Labels appear valid")
    return True


# ==============================================================================
# ROOT CAUSE 2: DO EMBEDDINGS CONTAIN DISCRIMINATIVE INFORMATION?
# ==============================================================================

def check_embedding_discriminability(text_emb, image_emb, meta_emb, labels):
    """Check if embeddings can separate classes"""
    print("\n" + "="*80)
    print("ROOT CAUSE 2: EMBEDDING DISCRIMINABILITY")
    print("="*80)
    
    embeddings = {
        'Text': text_emb,
        'Image': image_emb,
        'Metadata': meta_emb
    }
    
    results = {}
    
    for name, emb in embeddings.items():
        print(f"\n{name} embeddings:")
        
        # Convert to numpy safely
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy() if emb.requires_grad else emb.cpu().numpy()
        else:
            emb_np = np.array(emb)
        
        labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        
        # 1. Statistical test (t-test on means)
        real_emb = emb_np[labels_np == 0]
        fake_emb = emb_np[labels_np == 1]
        
        mean_real = real_emb.mean(axis=0)
        mean_fake = fake_emb.mean(axis=0)
        
        # T-test on each dimension
        significant_dims = 0
        for dim in range(emb_np.shape[1]):
            _, p_value = ttest_ind(real_emb[:, dim], fake_emb[:, dim])
            if p_value < 0.05:
                significant_dims += 1
        
        pct_significant = significant_dims / emb_np.shape[1] * 100
        print(f"  Statistically significant dimensions: {significant_dims}/{emb_np.shape[1]} ({pct_significant:.1f}%)")
        
        # 2. Simple classifier test
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            
            # Split for testing
            n_train = int(0.7 * len(labels_np))
            clf.fit(emb_np[:n_train], labels_np[:n_train])
            preds = clf.predict(emb_np[n_train:])
            
            f1 = f1_score(labels_np[n_train:], preds)
            acc = accuracy_score(labels_np[n_train:], preds)
            
            print(f"  Simple logistic regression: F1={f1:.3f}, Acc={acc:.3f}")
            
            if f1 < 0.55:
                print(f"  ❌ PROBLEM: {name} embeddings have weak discriminative power")
                results[name] = 'poor'
            elif f1 < 0.65:
                print(f"  ⚠️  WARNING: {name} embeddings have moderate discriminative power")
                results[name] = 'moderate'
            else:
                print(f"  ✅ GOOD: {name} embeddings have strong discriminative power")
                results[name] = 'good'
        
        except Exception as e:
            print(f"  ❌ ERROR: Could not train classifier - {e}")
            results[name] = 'error'
        
        # 3. Embedding statistics
        cos_sim = np.dot(mean_real, mean_fake) / (np.linalg.norm(mean_real) * np.linalg.norm(mean_fake))
        l2_dist = np.linalg.norm(mean_real - mean_fake)
        
        print(f"  Cosine similarity (real vs fake): {cos_sim:.4f}")
        print(f"  L2 distance (real vs fake): {l2_dist:.4f}")
        
        if cos_sim > 0.99 and l2_dist < 0.1:
            print(f"  ❌ CRITICAL: Embeddings are nearly identical!")
    
    return results


# ==============================================================================
# ROOT CAUSE 3: IS THE MODEL ARCHITECTURE APPROPRIATE?
# ==============================================================================

def check_model_capacity(model, sample_batch):
    """Check if model has enough capacity and if gradients flow"""
    print("\n" + "="*80)
    print("ROOT CAUSE 3: MODEL ARCHITECTURE CHECK")
    print("="*80)
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if trainable_params < 100000:
        print("⚠️  WARNING: Model might be too simple (< 100K parameters)")
    elif trainable_params > 50000000:
        print("⚠️  WARNING: Model might be too complex (> 50M parameters)")
    else:
        print("✅ Model capacity seems reasonable")
    
    # Check gradient flow
    print("\nChecking gradient flow...")
    model.train()
    h_text, h_image, h_meta, labels = sample_batch
    h_text = h_text.to(device).requires_grad_(True)
    h_image = h_image.to(device).requires_grad_(True)
    h_meta = h_meta.to(device).requires_grad_(True)
    labels = labels.to(device).float()
    
    # Forward pass
    try:
        logits, intermediates = model(h_text, h_image, h_meta, return_intermediates=True)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
        loss.backward()
        
        # Check if gradients exist
        layers_with_grads = 0
        layers_without_grads = 0
        vanishing_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                layers_with_grads += 1
                if param.grad.abs().mean() < 1e-7:
                    vanishing_grads += 1
            else:
                layers_without_grads += 1
        
        print(f"  Layers with gradients: {layers_with_grads}")
        print(f"  Layers without gradients: {layers_without_grads}")
        print(f"  Layers with vanishing gradients: {vanishing_grads}")
        
        if layers_without_grads > 0:
            print(f"  ❌ PROBLEM: {layers_without_grads} layers not receiving gradients!")
        elif vanishing_grads > layers_with_grads * 0.3:
            print(f"  ⚠️  WARNING: Many layers have vanishing gradients")
        else:
            print(f"  ✅ Gradients flowing properly")
        
    except Exception as e:
        print(f"  ❌ ERROR: Forward/backward pass failed - {e}")
        return False
    
    return True


# ==============================================================================
# ROOT CAUSE 4: ARE THERE DATA PREPROCESSING ISSUES?
# ==============================================================================

def check_preprocessing_issues(text_emb, image_emb, meta_emb):
    """Check for common preprocessing problems"""
    print("\n" + "="*80)
    print("ROOT CAUSE 4: PREPROCESSING CHECK")
    print("="*80)
    
    embeddings = {
        'Text': text_emb,
        'Image': image_emb,
        'Metadata': meta_emb
    }
    
    for name, emb in embeddings.items():
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy() if emb.requires_grad else emb.cpu().numpy()
        else:
            emb_np = emb
        
        print(f"\n{name} embeddings:")
        
        # Check for NaN/Inf
        has_nan = np.isnan(emb_np).any()
        has_inf = np.isinf(emb_np).any()
        
        if has_nan:
            print(f"  ❌ CRITICAL: Contains NaN values!")
        if has_inf:
            print(f"  ❌ CRITICAL: Contains Inf values!")
        
        if not has_nan and not has_inf:
            print(f"  ✅ No NaN/Inf values")
        
        # Check variance
        variance = emb_np.var(axis=0)
        low_variance_dims = sum(variance < 1e-6)
        
        if low_variance_dims > emb_np.shape[1] * 0.5:
            print(f"  ❌ PROBLEM: {low_variance_dims}/{emb_np.shape[1]} dimensions have very low variance")
        elif low_variance_dims > 0:
            print(f"  ⚠️  WARNING: {low_variance_dims} dimensions have low variance")
        else:
            print(f"  ✅ All dimensions have reasonable variance")
        
        # Check mean and std
        mean = emb_np.mean()
        std = emb_np.std()
        
        print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
        
        if abs(mean) > 10:
            print(f"  ⚠️  WARNING: Mean is far from zero - consider normalization")
        if std < 0.1 or std > 10:
            print(f"  ⚠️  WARNING: Unusual std deviation - check scaling")


# ==============================================================================
# ROOT CAUSE 5: TRAINING PROCEDURE ISSUES
# ==============================================================================

def check_training_setup(learning_rate=1e-4, batch_size=32, num_epochs=20):
    """Check if training hyperparameters are reasonable"""
    print("\n" + "="*80)
    print("ROOT CAUSE 5: TRAINING SETUP CHECK")
    print("="*80)
    
    print(f"Learning rate: {learning_rate}")
    if learning_rate > 1e-3:
        print("  ⚠️  WARNING: LR might be too high")
    elif learning_rate < 1e-5:
        print("  ⚠️  WARNING: LR might be too low")
    else:
        print("  ✅ LR seems reasonable")
    
    print(f"\nBatch size: {batch_size}")
    if batch_size < 16:
        print("  ⚠️  WARNING: Batch size might be too small")
    elif batch_size > 128:
        print("  ⚠️  WARNING: Batch size might be too large")
    else:
        print("  ✅ Batch size seems reasonable")
    
    print(f"\nNumber of epochs: {num_epochs}")
    if num_epochs < 10:
        print("  ⚠️  WARNING: Might not train long enough")
    else:
        print("  ✅ Epochs seem sufficient")


# ==============================================================================
# COMPLETE ROOT CAUSE ANALYSIS
# ==============================================================================

def run_complete_analysis():
    """Run all diagnostics"""
    print("\n" + "="*80)
    print("COMPLETE ROOT CAUSE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        # Use the corrected labels file
        df = pd.read_pickle("Dataset/twitter/df_with_corrected_labels.pkl")
        text_emb = torch.load("metadata_dense_embeddings.pt")
        image_emb = torch.load("metadata_user_sequence_embeddings.pt").squeeze(1)
        meta_emb = torch.load("metadata_user_sequence_embeddings.pt").squeeze(1)
        
        print(f"✅ Data loaded: {len(df)} samples")
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return
    
    # 1. Check labels
    labels_ok = check_labels(df)
    
    if not labels_ok:
        print("\n" + "="*80)
        print("CRITICAL: FIX LABELS FIRST!")
        print("="*80)
        print("\nYour DataFrame doesn't have valid labels.")
        print("Run the label fixer script first:")
        print("  python label_fixer.py")
        return
    
    # Safe label conversion
    try:
        # Labels are already fixed as integers
        labels = torch.tensor(df['label'].values, dtype=torch.float32)
        print(f"✅ Successfully loaded {len(labels)} labels")
        print(f"   Real (0): {sum(labels==0).item()}, Fake (1): {sum(labels==1).item()}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not convert labels to tensor: {e}")
        return
    
    # 2. Check embeddings
    emb_results = check_embedding_discriminability(text_emb, image_emb, meta_emb, labels)
    
    # 3. Check model
    from multimodal_fakenews_model import AdaptiveMultimodalFakeNewsDetector
    model = AdaptiveMultimodalFakeNewsDetector(
        d_text=128,
        d_image=128,
        d_meta=128,
        d_common=256
    ).to(device)
    
    # Create sample batch
    sample_batch = (
        text_emb[:8].to(device),
        image_emb[:8].to(device),
        meta_emb[:8].to(device),
        labels[:8].to(device)
    )
    
    model_ok = check_model_capacity(model, sample_batch)
    
    # 4. Check preprocessing
    check_preprocessing_issues(text_emb, image_emb, meta_emb)
    
    # 5. Check training setup
    check_training_setup()
    
    # Final summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    all_emb_poor = all(v == 'poor' for v in emb_results.values())
    some_emb_poor = any(v == 'poor' for v in emb_results.values())
    
    if all_emb_poor:
        print("\n❌ PRIMARY ISSUE: Embeddings have no discriminative power")
        print("   → This means your text/image/metadata features cannot distinguish real from fake")
        print("   → Solutions:")
        print("     1. Use better pre-trained models (larger BERT, better vision models)")
        print("     2. Fine-tune embeddings on your domain")
        print("     3. Add more informative features")
    elif some_emb_poor:
        print("\n⚠️  PARTIAL ISSUE: Some embeddings are weak")
        print(f"   → Weak modalities: {[k for k, v in emb_results.items() if v == 'poor']}")
        print("   → Consider removing weak modalities or improving their quality")
    else:
        print("\n✅ Embeddings have discriminative power")
        print("   → Issue is likely in model architecture or training procedure")
        print("   → Try the improved training script with:")
        print("     1. Better loss function (focal loss)")
        print("     2. Data augmentation")
        print("     3. Learning rate warmup")
        print("     4. Longer training with early stopping")

if __name__ == "__main__":
    run_complete_analysis()