import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print("\n[STEP 1] Loading data...")

try:
    prepared_data = torch.load("prepared_clustering_data.pt", map_location=device, weights_only=False)
    z_out = prepared_data['z_out'].cpu().numpy()
    v_mismatch = prepared_data['v_mismatch'].cpu().numpy()
    user_ids = prepared_data['user_ids']
    timestamps = prepared_data['timestamps']
    post_ids = prepared_data['post_ids']
    
    print(f" Loaded: {len(z_out)} posts")
except:
    print(" Could not load prepared data")
    exit(1)
    
# Load ground truth labels
print("Loading ground truth labels...")
df_labels = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df_labels = df_labels.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)

# Align labels to post_ids in prepared_data
post_id_to_label = dict(zip(df_labels['post_id'], df_labels['label']))
labels_aligned = np.array([
    post_id_to_label.get(pid, 'unknown') for pid in post_ids
])

real_mask = labels_aligned == 'real'
print(f" Real posts: {real_mask.sum()} | Fake posts: {(~real_mask).sum()}")

print("\n[STEP 2] Advanced feature engineering...")

# Mismatch features
mismatch_magnitude = np.linalg.norm(v_mismatch, axis=1, keepdims=True)

# Statistical features
z_abs = np.abs(z_out)
z_abs_normalized = z_abs / (z_abs.sum(axis=1, keepdims=True) + 1e-10)
z_entropy = -np.sum(z_abs_normalized * np.log(z_abs_normalized + 1e-10), axis=1, keepdims=True)
z_variance = np.var(z_out, axis=1, keepdims=True)
z_skewness = np.mean((z_out - z_out.mean(axis=1, keepdims=True))**3, axis=1, keepdims=True)
z_kurtosis = np.mean((z_out - z_out.mean(axis=1, keepdims=True))**4, axis=1, keepdims=True)

# Cross-modal features
z_v_dot = np.sum(z_out * v_mismatch, axis=1, keepdims=True)
z_v_cosine = z_v_dot / (np.linalg.norm(z_out, axis=1, keepdims=True) * 
                         np.linalg.norm(v_mismatch, axis=1, keepdims=True) + 1e-10)

# Concatenate
X_engineered = np.hstack([
    z_out,                    # 128-dim
    v_mismatch,               # 128-dim
    mismatch_magnitude,       # 1-dim
    z_entropy,                # 1-dim
    z_variance,               # 1-dim
    z_skewness,               # 1-dim
    z_kurtosis,               # 1-dim
    z_v_cosine                # 1-dim
])

print(f" Feature vector: {X_engineered.shape}")

print("\n[STEP 3] Scaling & dimensionality reduction...")

# Robust scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_engineered)

# PCA (95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X_scaled)

print(f"\ Reduced: {X_scaled.shape[1]} → {X_reduced.shape[1]} dimensions")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")


print("\n[STEP 4] Running multiple anomaly detection algorithms...")
print("\n[STEP 4] Running multiple anomaly detection algorithms...")
print(f"   Fitting on {real_mask.sum()} real posts, scoring all {len(X_reduced)} posts...")

anomaly_scores = {}

# 4.1 Isolation Forest
print("   [4.1] Isolation Forest...")
iso_forest = IsolationForest(
    contamination=0.1,
    random_state=42,
    n_estimators=200,
    n_jobs=-1
)
iso_forest.fit(X_reduced[real_mask])
iso_scores = -iso_forest.score_samples(X_reduced)
iso_labels = iso_forest.predict(X_reduced)
anomaly_scores['isolation_forest'] = iso_scores
print(f"    Detected {(iso_labels == -1).sum()} anomalies ({(iso_labels == -1).sum()/len(X_reduced)*100:.1f}%)")

# 4.2 Local Outlier Factor
print("   [4.2] Local Outlier Factor...")
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=True
)
lof.fit(X_reduced[real_mask])
lof_labels = lof.predict(X_reduced)
lof_scores = -lof.score_samples(X_reduced)
anomaly_scores['lof'] = lof_scores
print(f"    Detected {(lof_labels == -1).sum()} anomalies ({(lof_labels == -1).sum()/len(X_reduced)*100:.1f}%)")

# 4.3 One-Class SVM
print("   [4.3] One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
ocsvm.fit(X_reduced[real_mask])
ocsvm_labels = ocsvm.predict(X_reduced)

ocsvm_raw = -ocsvm.score_samples(X_reduced)  # all negative
ocsvm_shift = float(ocsvm_raw.min())         # save this offset
ocsvm_scores = ocsvm_raw - ocsvm_shift       # shift to [0, range]
anomaly_scores['ocsvm'] = ocsvm_scores
print(f"    Detected {(ocsvm_labels == -1).sum()} anomalies ({(ocsvm_labels == -1).sum()/len(X_reduced)*100:.1f}%)")

# 4.4 Elliptic Envelope
print("   [4.4] Elliptic Envelope (Robust Covariance)...")
elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
elliptic.fit(X_reduced[real_mask])
elliptic_labels = elliptic.predict(X_reduced)
elliptic_scores = -elliptic.score_samples(X_reduced)
anomaly_scores['elliptic'] = elliptic_scores
print(f"    Detected {(elliptic_labels == -1).sum()} anomalies ({(elliptic_labels == -1).sum()/len(X_reduced)*100:.1f}%)")

print("\n[STEP 5] Creating ensemble anomaly scores...")

# Normalize all scores to [0, 1]
anomaly_scores_normalized = {}
for method, scores in anomaly_scores.items():
    min_score = scores.min()
    max_score = scores.max()
    normalized = (scores - min_score) / (max_score - min_score + 1e-10)
    anomaly_scores_normalized[method] = normalized

# Ensemble: weighted average
weights = {
    'isolation_forest': 0.35,  
    'lof': 0.30,               
    'ocsvm': 0.20,             
    'elliptic': 0.15           
}

ensemble_score = np.zeros(len(X_reduced))
for method, weight in weights.items():
    ensemble_score += weight * anomaly_scores_normalized[method]

# Define anomaly levels based on ensemble score
anomaly_percentiles = np.percentile(ensemble_score, [75, 90, 95, 99])

def get_anomaly_level(score):
    if score >= anomaly_percentiles[3]:
        return 'critical'
    elif score >= anomaly_percentiles[2]:
        return 'high'
    elif score >= anomaly_percentiles[1]:
        return 'medium'
    elif score >= anomaly_percentiles[0]:
        return 'low'
    else:
        return 'normal'

anomaly_levels = np.array([get_anomaly_level(s) for s in ensemble_score])

# Convert to numeric labels for compatibility with clustering output format
level_to_id = {'normal': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
anomaly_labels = np.array([level_to_id[level] for level in anomaly_levels])

print(f" Ensemble scoring complete")
print(f"\n   Anomaly Distribution:")
for level in ['normal', 'low', 'medium', 'high', 'critical']:
    count = (anomaly_levels == level).sum()
    pct = count / len(anomaly_levels) * 100
    print(f"      {level:8s}: {count:5d} posts ({pct:5.1f}%)")

print("\n[STEP 7] Enhanced user behavior analysis...")

# Create results dataframe
results_df = pd.DataFrame({
    'post_id': post_ids,
    'user_id': user_ids,
    'timestamp': timestamps,
    'anomaly_score': ensemble_score,
    'anomaly_level': anomaly_levels,
    'anomaly_label': anomaly_labels,
    'iso_forest_score': anomaly_scores_normalized['isolation_forest'],
    'lof_score': anomaly_scores_normalized['lof'],
    'ocsvm_score': anomaly_scores_normalized['ocsvm'],
    'elliptic_score': anomaly_scores_normalized['elliptic']
})

user_behavior = {}

for user_id in tqdm(set(results_df['user_id']), desc="Analyzing users"):
    user_posts = results_df[results_df['user_id'] == user_id]
    n_posts = len(user_posts)
    
    # Anomaly statistics
    mean_anomaly_score = user_posts['anomaly_score'].mean()
    max_anomaly_score = user_posts['anomaly_score'].max()
    std_anomaly_score = user_posts['anomaly_score'].std()
    
    # Count by severity
    n_critical = (user_posts['anomaly_level'] == 'critical').sum()
    n_high = (user_posts['anomaly_level'] == 'high').sum()
    n_medium = (user_posts['anomaly_level'] == 'medium').sum()
    n_low = (user_posts['anomaly_level'] == 'low').sum()
    n_normal = (user_posts['anomaly_level'] == 'normal').sum()
    
    # Calculate risk score (0-100)
    risk_score = (
        n_critical * 10 +
        n_high * 5 +
        n_medium * 2 +
        n_low * 0.5
    ) / max(n_posts, 1) * 10 * min(n_posts/ 5, 1.0)
    
    risk_score = min(risk_score, 100)  # Cap at 100
    
    # Determine suspicion level
    is_suspicious = (
        (n_critical >= 2) or
        (n_high >= 3) or
        (mean_anomaly_score > anomaly_percentiles[2]) or
        (risk_score > 20)
    )
    
    # Risk category
    if risk_score >= 35:
        risk_category = 'critical'
    elif risk_score >= 20:
        risk_category = 'high'
    elif risk_score >= 10:
        risk_category = 'medium'
    else:
        risk_category = 'low'
    
    user_behavior[user_id] = {
        'n_posts': int(n_posts),
        'mean_anomaly_score': float(mean_anomaly_score),
        'max_anomaly_score': float(max_anomaly_score),
        'std_anomaly_score': float(std_anomaly_score),
        'n_critical': int(n_critical),
        'n_high': int(n_high),
        'n_medium': int(n_medium),
        'n_low': int(n_low),
        'n_normal': int(n_normal),
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'is_suspicious': bool(is_suspicious)
    }

suspicious_users = {
    uid: behavior for uid, behavior in user_behavior.items()
    if behavior['is_suspicious']
}

print(f"\n User Analysis Complete:")
print(f"   Total users: {len(user_behavior)}")
print(f"   Suspicious users: {len(suspicious_users)} ({len(suspicious_users)/len(user_behavior)*100:.1f}%)")

if suspicious_users:
    print(f"\n   Top 10 Most Suspicious Users:")
    for user_id, behavior in sorted(suspicious_users.items(), 
                                    key=lambda x: x[1]['risk_score'], 
                                    reverse=True)[:10]:
        print(f"      {user_id}: risk={behavior['risk_score']:.1f}/100 | "
              f"posts={behavior['n_posts']} | "
              f"critical={behavior['n_critical']} high={behavior['n_high']} medium={behavior['n_medium']}")


print("\n[STEP 8] Saving results...")

output_dir = Path("anomaly_detection_results")
output_dir.mkdir(exist_ok=True)

# Save main results (compatible with clustering format)
results_df.to_csv(output_dir / "anomaly_assignments.csv", index=False)

# Save user analysis
user_df = pd.DataFrame.from_dict(user_behavior, orient='index')
user_df.index.name = 'user_id'
user_df = user_df.sort_values('risk_score', ascending=False)
user_df.to_csv(output_dir / "user_risk_analysis.csv")

# Save suspicious users separately
if suspicious_users:
    suspicious_df = pd.DataFrame.from_dict(suspicious_users, orient='index')
    suspicious_df.index.name = 'user_id'
    suspicious_df = suspicious_df.sort_values('risk_score', ascending=False)
    suspicious_df.to_csv(output_dir / "suspicious_users.csv")

# Save per-method score distributions for correct inference normalization
method_score_distributions = {}
for name, (scores_raw, scores_norm) in [
    ('isolation_forest', (anomaly_scores['isolation_forest'], anomaly_scores_normalized['isolation_forest'])),
    ('lof',              (anomaly_scores['lof'],              anomaly_scores_normalized['lof'])),
    ('ocsvm',            (anomaly_scores['ocsvm'],            anomaly_scores_normalized['ocsvm'])),
    ('elliptic',         (anomaly_scores['elliptic'],         anomaly_scores_normalized['elliptic'])),
]:
    method_score_distributions[name] = {
        'min':  float(scores_raw.min()),
        'max':  float(scores_raw.max()),
        'mean': float(scores_raw.mean()),
        'std':  float(scores_raw.std()),
        'p25':  float(np.percentile(scores_raw, 25)),
        'p75':  float(np.percentile(scores_raw, 75)),
        'p95':  float(np.percentile(scores_raw, 95)),
    }
    print(f"  {name}: raw range=[{scores_raw.min():.4f}, {scores_raw.max():.4f}]")

ensemble_score_distribution = {
    'min':  float(ensemble_score.min()),
    'max':  float(ensemble_score.max()),
    'mean': float(ensemble_score.mean()),
    'std':  float(ensemble_score.std()),
    'p25':  float(np.percentile(ensemble_score, 25)),
    'p50':  float(np.percentile(ensemble_score, 50)),
    'p75':  float(np.percentile(ensemble_score, 75)),
    'p90':  float(np.percentile(ensemble_score, 90)),
    'p95':  float(np.percentile(ensemble_score, 95)),
    'p99':  float(np.percentile(ensemble_score, 99)),
}

torch.save({
    'scaler': scaler,
    'pca': pca,
    
    'models': {
        'isolation_forest': iso_forest,
        'lof': lof,
        'ocsvm': ocsvm,
        'elliptic': elliptic
    },
    'ensemble_weights': weights,
    'anomaly_percentiles': anomaly_percentiles,
    'method_score_distributions': method_score_distributions,
    'ensemble_score_distribution': ensemble_score_distribution,
    'X_reduced': X_reduced,
    'ocsvm_shift': ocsvm_shift
}, output_dir / "anomaly_models.pt")

print(f"\n Saved with score distributions")
print(f"   Ensemble: p25={ensemble_score_distribution['p25']:.4f} "
      f"p50={ensemble_score_distribution['p50']:.4f} "
      f"p75={ensemble_score_distribution['p75']:.4f} "
      f"p95={ensemble_score_distribution['p95']:.4f}")

print(f" Saved results to {output_dir}/")

print("\n[STEP 9] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')

# 1. Anomaly score distribution
axes[0, 0].hist(ensemble_score, bins=50, color='coral', edgecolor='black', alpha=0.7)
for i, pct in enumerate([75, 90, 95, 99]):
    axes[0, 0].axvline(anomaly_percentiles[i], color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'{pct}th percentile')
axes[0, 0].set_title('Anomaly Score Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Ensemble Anomaly Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend(fontsize=8)

# 2. Method comparison
method_counts = {
    'IsoForest': (iso_labels == -1).sum(),
    'LOF': (lof_labels == -1).sum(),
    'OCSVM': (ocsvm_labels == -1).sum(),
    'Elliptic': (elliptic_labels == -1).sum()
}
axes[0, 1].bar(method_counts.keys(), method_counts.values(),
               color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], edgecolor='black')
axes[0, 1].set_title('Anomalies Detected by Method', fontweight='bold')
axes[0, 1].set_ylabel('Number of Anomalies')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Severity distribution
level_counts = {level: (anomaly_levels == level).sum()
                for level in ['normal', 'low', 'medium', 'high', 'critical']}
colors_pie = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
axes[1, 0].pie(level_counts.values(), labels=level_counts.keys(), autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
axes[1, 0].set_title('Anomaly Severity Distribution', fontweight='bold')

# 4. User risk distribution
user_risk_counts = {}
for b in user_behavior.values():
    risk = b['risk_category']
    user_risk_counts[risk] = user_risk_counts.get(risk, 0) + 1

risk_order = ['low', 'medium', 'high', 'critical']
risk_colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
risk_values = [user_risk_counts.get(r, 0) for r in risk_order]
axes[1, 1].bar(risk_order, risk_values, color=risk_colors, edgecolor='black')
axes[1, 1].set_title('User Risk Categories', fontweight='bold')
axes[1, 1].set_ylabel('Number of Users')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "anomaly_analysis.png", dpi=150, bbox_inches='tight')
print(f"\ Saved visualization")