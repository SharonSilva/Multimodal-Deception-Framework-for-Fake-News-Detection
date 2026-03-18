import pandas as pd
import numpy as np
from scipy import stats

# Load your preprocessed dataset
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")

# Separate fake and real
fake_scores = df[df['label']=='fake']['anomaly_score']
real_scores = df[df['label']=='real']['anomaly_score']

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(fake_scores, real_scores, alternative='greater')

# Cohen's d
d = (fake_scores.mean() - real_scores.mean()) / np.sqrt(
    (fake_scores.std()**2 + real_scores.std()**2) / 2
)

print(f"Fake mean: {fake_scores.mean():.4f}")
print(f"Real mean: {real_scores.mean():.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Cohen's d: {d:.4f}")