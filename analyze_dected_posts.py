""""

Compare unsupervised detections with original data to understand what was found.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Load detection results
detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
detections['post_id'] = detections['post_id'].astype(int)  # Ensure int type

# Load original data
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(int)  # Ensure int type

print(f"Available columns in dataset: {list(df.columns)[:10]}...")

# Load anomaly results
anomaly_results = pd.read_csv("anomaly_detection_results/anomaly_assignments.csv")
anomaly_results['post_id'] = anomaly_results['post_id'].astype(int)  # Ensure int type

# Merge with flexible column names
text_col = 'post_text' if 'post_text' in df.columns else ('text' if 'text' in df.columns else 'tweet')
user_col = 'username' if 'username' in df.columns else 'user_id'
label_col = 'label' if 'label' in df.columns else 'class'

# Build merge column list
df_cols = ['post_id', text_col, user_col, label_col]
if 'timestamp' in df.columns:
    df_cols.append('timestamp')

# Merge everything
merged = detections.merge(
    anomaly_results[['post_id', 'anomaly_score', 'anomaly_label']], 
    on='post_id', 
    how='left'
)

merged = merged.merge(
    df[df_cols], 
    on='post_id', 
    how='left'
)

# Standardize column names
if text_col != 'text':
    merged['text'] = merged[text_col]
if user_col != 'username':
    merged['username'] = merged[user_col]

print(f"\nLoaded {len(merged)} posts with detections\n")


# ANALYSIS 1: What did the model flag as suspicious?

print("="*80)
print("ANALYSIS 1: MODEL DETECTIONS")
print("="*80)

suspicious = merged[merged['is_suspicious'] == True]
normal = merged[merged['is_suspicious'] == False]

print(f"\nDetected as SUSPICIOUS: {len(suspicious)} posts")
print(f"Detected as NORMAL: {len(normal)} posts")

print(f"\nSuspicion Score Statistics:")
print(f"   Suspicious posts - Mean score: {suspicious['suspicion_score'].mean():.4f}")
print(f"   Normal posts - Mean score: {normal['suspicion_score'].mean():.4f}")

print(f"\nOriginal Anomaly Scores (from your data):")
print(f"   Suspicious posts - Mean anomaly: {suspicious['anomaly_score'].mean():.4f}")
print(f"   Normal posts - Mean anomaly: {normal['anomaly_score'].mean():.4f}")


# ANALYSIS 2: Compare with different anomaly thresholds

print("\n" + "="*80)
print("ANALYSIS 2: PERFORMANCE AT DIFFERENT THRESHOLDS")
print("="*80)

print("\nLet's see how detection performs with different 'ground truth' thresholds:")
print("-"*80)

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for thresh in thresholds:
    # Define ground truth at this threshold
    true_suspicious = merged['anomaly_score'] > thresh
    detected_suspicious = merged['is_suspicious'] == True
    
    tp = (detected_suspicious & true_suspicious).sum()
    fp = (detected_suspicious & ~true_suspicious).sum()
    fn = (~detected_suspicious & true_suspicious).sum()
    tn = (~detected_suspicious & ~true_suspicious).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    num_true_suspicious = true_suspicious.sum()
    
    print(f"\nThreshold {thresh:.1f} ({num_true_suspicious} suspicious in ground truth):")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")


# ANALYSIS 3: Check actual labels (fake vs real)

print("\n" + "="*80)
print("ANALYSIS 3: DETECTED POSTS vs TRUE LABELS (FAKE/REAL)")
print("="*80)

print("\nLet's check if detected posts are actually FAKE news:")

# Map label to binary (assuming 'label' column has fake/real)
if 'label' in merged.columns:
    suspicious_posts = merged[merged['is_suspicious'] == True]
    
    # Count fake vs real in detected posts
    if merged['label'].dtype == 'object':
        fake_detected = (suspicious_posts['label'].str.lower() == 'fake').sum()
        real_detected = (suspicious_posts['label'].str.lower() == 'real').sum()
    else:
        fake_detected = (suspicious_posts['label'] == 1).sum()
        real_detected = (suspicious_posts['label'] == 0).sum()
    
    total_detected = len(suspicious_posts)
    
    print(f"\n Among {total_detected} detected suspicious posts:")
    print(f"   FAKE news: {fake_detected} ({fake_detected/total_detected*100:.1f}%)")
    print(f"   REAL news: {real_detected} ({real_detected/total_detected*100:.1f}%)")
    
    # Overall dataset distribution
    if merged['label'].dtype == 'object':
        total_fake = (merged['label'].str.lower() == 'fake').sum()
        total_real = (merged['label'].str.lower() == 'real').sum()
    else:
        total_fake = (merged['label'] == 1).sum()
        total_real = (merged['label'] == 0).sum()
    
    print(f"\n Overall dataset:")
    print(f"   FAKE news: {total_fake} ({total_fake/len(merged)*100:.1f}%)")
    print(f"   REAL news: {total_real} ({total_real/len(merged)*100:.1f}%)")
    
    # Detection rate
    fake_detection_rate = fake_detected / total_fake if total_fake > 0 else 0
    real_detection_rate = real_detected / total_real if total_real > 0 else 0
    
    print(f"\n Detection rates:")
    print(f"   Caught {fake_detection_rate*100:.1f}% of fake news")
    print(f"   Flagged {real_detection_rate*100:.1f}% of real news (false positives)")


# ANALYSIS 4: Top suspicious posts

print("\n" + "="*80)
print("ANALYSIS 4: TOP 20 MOST SUSPICIOUS POSTS")
print("="*80)

top_suspicious = merged.nlargest(20, 'suspicion_score')

print("\nPost ID | Suspicion | Anomaly | Label | Text Preview")
print("-"*80)

for idx, row in top_suspicious.iterrows():
    post_id = row['post_id']
    susp_score = row['suspicion_score']
    anom_score = row['anomaly_score']
    label = str(row.get('label', 'unknown'))
    text = str(row.get('text', ''))[:50] + "..."
    
    print(f"{post_id:8d} | {susp_score:.4f}    | {anom_score:.4f}  | {label:7s} | {text}")


# ANALYSIS 5: Detection method breakdown

print("\n" + "="*80)
print("ANALYSIS 5: WHICH METHODS FLAGGED THESE POSTS?")
print("="*80)

suspicious = merged[merged['is_suspicious'] == True]

iso_only = (suspicious['iso_forest_flag'] == 1) & (suspicious['dbscan_outlier'] == 0) & (suspicious['high_distance'] == 0) & (suspicious['in_deception_cluster'] == 0)
dbscan_only = (suspicious['iso_forest_flag'] == 0) & (suspicious['dbscan_outlier'] == 1) & (suspicious['high_distance'] == 0) & (suspicious['in_deception_cluster'] == 0)
distance_only = (suspicious['iso_forest_flag'] == 0) & (suspicious['dbscan_outlier'] == 0) & (suspicious['high_distance'] == 1) & (suspicious['in_deception_cluster'] == 0)
deception_only = (suspicious['iso_forest_flag'] == 0) & (suspicious['dbscan_outlier'] == 0) & (suspicious['high_distance'] == 0) & (suspicious['in_deception_cluster'] == 1)

multiple_methods = (
    (suspicious['iso_forest_flag'] + suspicious['dbscan_outlier'] + 
     suspicious['high_distance'] + suspicious['in_deception_cluster']) >= 2
)

print(f"\n Detection method breakdown:")
print(f"   Isolation Forest only: {iso_only.sum()}")
print(f"   DBSCAN only: {dbscan_only.sum()}")
print(f"   High distance only: {distance_only.sum()}")
print(f"   Deception cluster only: {deception_only.sum()}")
print(f"   Multiple methods: {multiple_methods.sum()}")

# High confidence 
high_confidence = (
    (suspicious['iso_forest_flag'] + suspicious['dbscan_outlier'] + 
     suspicious['high_distance'] + suspicious['in_deception_cluster']) >= 3
)

print(f"\nHigh confidence detections (3+ methods agree): {high_confidence.sum()}")


# VISUALIZATION

print("\n Creating detailed analysis plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Suspicion score vs anomaly score
ax = axes[0, 0]
ax.scatter(merged['anomaly_score'], merged['suspicion_score'], 
           alpha=0.3, s=20, c=merged['is_suspicious'], cmap='RdYlGn_r')
ax.set_xlabel('Original Anomaly Score')
ax.set_ylabel('Unsupervised Suspicion Score')
ax.set_title('Suspicion vs Anomaly Scores')
ax.grid(True, alpha=0.3)

# Distribution comparison
ax = axes[0, 1]
ax.hist(suspicious['anomaly_score'], bins=30, alpha=0.7, label='Detected', color='red')
ax.hist(normal['anomaly_score'], bins=30, alpha=0.7, label='Normal', color='blue')
ax.set_xlabel('Original Anomaly Score')
ax.set_ylabel('Frequency')
ax.set_title('Anomaly Score Distribution')
ax.legend()
ax.set_yscale('log')

# F1 at different thresholds
ax = axes[0, 2]
f1_scores = []
for thresh in np.linspace(0.1, 0.9, 50):
    true_suspicious = merged['anomaly_score'] > thresh
    detected_suspicious = merged['is_suspicious'] == True
    tp = (detected_suspicious & true_suspicious).sum()
    fp = (detected_suspicious & ~true_suspicious).sum()
    fn = (~detected_suspicious & true_suspicious).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    f1_scores.append(f1)

ax.plot(np.linspace(0.1, 0.9, 50), f1_scores, linewidth=2)
ax.set_xlabel('Anomaly Score Threshold')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Ground Truth Threshold')
ax.grid(True, alpha=0.3)
best_thresh = np.linspace(0.1, 0.9, 50)[np.argmax(f1_scores)]
ax.axvline(best_thresh, color='red', linestyle='--', label=f'Best: {best_thresh:.2f}')
ax.legend()

# Label distribution (if available)
ax = axes[1, 0]
if 'label' in merged.columns:
    detected_labels = suspicious['label'].value_counts()
    ax.bar(range(len(detected_labels)), detected_labels.values, 
           color=['red' if 'fake' in str(l).lower() or l == 1 else 'green' 
                  for l in detected_labels.index])
    ax.set_xticks(range(len(detected_labels)))
    ax.set_xticklabels(detected_labels.index, rotation=45)
    ax.set_ylabel('Count')
    ax.set_title('Labels of Detected Suspicious Posts')
    ax.grid(axis='y', alpha=0.3)

# Method agreement
ax = axes[1, 1]
agreement_counts = (
    suspicious['iso_forest_flag'] + suspicious['dbscan_outlier'] + 
    suspicious['high_distance'] + suspicious['in_deception_cluster']
).value_counts().sort_index()

ax.bar(agreement_counts.index, agreement_counts.values, 
       color=['orange', 'yellow', 'lightgreen', 'green', 'darkgreen'][:len(agreement_counts)])
ax.set_xlabel('Number of Methods in Agreement')
ax.set_ylabel('Number of Posts')
ax.set_title('Detection Method Agreement')
ax.grid(axis='y', alpha=0.3)

# Precision/Recall curve
ax = axes[1, 2]
precisions = []
recalls = []
thresholds_list = np.linspace(0.1, 0.9, 50)

for thresh in thresholds_list:
    true_suspicious = merged['anomaly_score'] > thresh
    detected_suspicious = merged['is_suspicious'] == True
    tp = (detected_suspicious & true_suspicious).sum()
    fp = (detected_suspicious & ~true_suspicious).sum()
    fn = (~detected_suspicious & true_suspicious).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    precisions.append(precision)
    recalls.append(recall)

ax.plot(recalls, precisions, linewidth=2, marker='o', markersize=3)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('suspicious_detection_results/detailed_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved detailed analysis to suspicious_detection_results/detailed_analysis.png")


# SAVE HIGH CONFIDENCE DETECTIONS

high_conf_detections = suspicious[high_confidence].copy()
high_conf_detections = high_conf_detections.sort_values('suspicion_score', ascending=False)

high_conf_detections.to_csv(
    'suspicious_detection_results/high_confidence_suspicious.csv',
    index=False
)

print(f"\nSaved {len(high_conf_detections)} high-confidence detections")

print("\n" + "="*80)
print("="*80)
print(f"   1. Best anomaly threshold: {best_thresh:.2f} (not 0.5!)")
print(f"   2. High confidence detections: {high_confidence.sum()}")
print(f"   3. Check suspicious_detection_results/high_confidence_suspicious.csv")
print(f"   4. View suspicious_detection_results/detailed_analysis.png for plots")

