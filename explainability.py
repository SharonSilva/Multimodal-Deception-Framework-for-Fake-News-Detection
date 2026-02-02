# """
# EXPLAINABILITY LAYER FOR SUSPICIOUS CONTENT DETECTION
# ======================================================
# Provides interpretable explanations for why posts are flagged as suspicious.

# Components:
# 1. Text Attribution - Highlights suspicious words/phrases
# 2. Graph Explanation - Shows influence and spread patterns
# 3. Metadata Analysis - Account-level risk indicators
# 4. Campaign Detection - Coordination patterns
# 5. Interactive Visualizations - Analyst-friendly dashboards
# """

# import torch
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from collections import defaultdict
# import networkx as nx
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("="*80)
# print("EXPLAINABILITY LAYER FOR SUSPICIOUS CONTENT DETECTION")
# print("="*80)

# # ============================================================================
# # LOAD DETECTION RESULTS & DATA
# # ============================================================================
# print("\n📦 Loading detection results and data...")

# # Load detection results
# detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
# high_conf = pd.read_csv("suspicious_detection_results/high_confidence_suspicious.csv")

# # Load original data
# df = pd.read_pickle("Dataset/twitter/df_with_all_features.pkl")
# df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
# df['post_id'] = df['post_id'].astype(int)

# # Merge
# merged = detections.merge(df, on='post_id', how='left')

# print(f"✅ Loaded {len(detections)} detections")
# print(f"✅ Loaded {len(high_conf)} high-confidence detections")

# # ============================================================================
# # 1. TEXT ATTRIBUTION - HIGHLIGHT SUSPICIOUS PHRASES
# # ============================================================================
# print("\n[1] Computing Text Attribution Scores...")

# # Load pre-trained model for text attribution
# node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# # Normalize features
# for ntype, features in node_features.items():
#     features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
#     node_features[ntype] = F.normalize(features, p=2, dim=1).to(device)

# edge_dict = {k: (ei.to(device), ew.to(device)) for k, (ei, ew) in edge_dict.items()}

# # Simple text attribution based on TF-IDF and suspicion correlation
# def compute_text_attribution(text_series, suspicion_scores):
#     """
#     Compute word importance scores based on correlation with suspicion.
#     Returns dict of {word: importance_score}
#     """
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from scipy.stats import spearmanr
    
#     # Clean text
#     texts = text_series.fillna('').astype(str).tolist()
    
#     # TF-IDF vectorization
#     vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     feature_names = vectorizer.get_feature_names_out()
    
#     # Compute correlation with suspicion scores
#     word_scores = {}
#     for i, word in enumerate(feature_names):
#         word_tfidf = tfidf_matrix[:, i].toarray().flatten()
#         corr, _ = spearmanr(word_tfidf, suspicion_scores)
#         if not np.isnan(corr):
#             word_scores[word] = abs(corr)
    
#     return word_scores, vectorizer

# # Compute for detected posts
# text_col = 'post_text' if 'post_text' in merged.columns else 'text'
# word_scores, vectorizer = compute_text_attribution(
#     merged[text_col], 
#     merged['suspicion_score']
# )

# # Get top suspicious phrases
# top_phrases = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:30]

# print(f"\n📊 Top Suspicious Phrases:")
# for phrase, score in top_phrases[:10]:
#     print(f"   '{phrase}': {score:.4f}")

# # ============================================================================
# # 2. GRAPH EXPLANATION - INFLUENCE & SPREAD PATTERNS
# # ============================================================================
# print("\n[2] Analyzing Graph Influence Patterns...")

# # Load graph structure
# import pickle
# with open("heterogeneous_graph/edges.pkl", "rb") as f:
#     edges = pickle.load(f)

# # Build influence network for suspicious posts
# suspicious_posts = set(high_conf['post_id'].values)

# # Get user-post connections
# user_post_edges = edges.get(('user', 'creates', 'post'), [])

# # Build influence graph
# influence_graph = nx.DiGraph()

# for src, dst, weight, timestamp in user_post_edges:
#     # Map to original IDs
#     user_id = [uid for uid, idx in node_mappings['user'].items() if idx == src]
#     post_id = [pid for pid, idx in node_mappings['post'].items() if idx == dst]
    
#     if user_id and post_id:
#         user_id = user_id[0]
#         post_id = post_id[0]
        
#         if post_id in suspicious_posts:
#             influence_graph.add_edge(user_id, post_id, weight=weight, timestamp=timestamp)

# print(f"✅ Built influence graph: {len(influence_graph.nodes)} nodes, {len(influence_graph.edges)} edges")

# # Compute centrality metrics
# if len(influence_graph.nodes) > 0:
#     try:
#         pagerank = nx.pagerank(influence_graph)
#         betweenness = nx.betweenness_centrality(influence_graph)
        
#         # Top influential users
#         top_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
#         print(f"\n📊 Top Influential Users in Suspicious Network:")
#         for user, score in top_users:
#             print(f"   User {user}: PageRank={score:.4f}")
#     except:
#         print("   ⚠️ Could not compute centrality (graph too small)")

# # ============================================================================
# # 3. METADATA ANALYSIS - ACCOUNT RISK INDICATORS
# # ============================================================================
# print("\n[3] Computing Account Risk Indicators...")

# # Compute user-level metadata features
# user_risk_scores = {}

# for user_id in merged['username'].unique():
#     user_posts = merged[merged['username'] == user_id]
    
#     if len(user_posts) == 0:
#         continue
    
#     # Compute risk indicators
#     posting_frequency = len(user_posts)
#     avg_suspicion = user_posts['suspicion_score'].mean()
#     high_conf_posts = (user_posts['suspicion_score'] > 0.5).sum()
    
#     # Temporal pattern (burstiness)
#     if 'timestamp' in user_posts.columns:
#         times = pd.to_datetime(user_posts['timestamp']).sort_values()
#         if len(times) > 1:
#             time_diffs = times.diff().dt.total_seconds().dropna()
#             burstiness = time_diffs.std() / (time_diffs.mean() + 1e-10) if len(time_diffs) > 0 else 0
#         else:
#             burstiness = 0
#     else:
#         burstiness = 0
    
#     # Hashtag diversity
#     if 'hashtags' in user_posts.columns:
#         all_hashtags = [h for hashtags in user_posts['hashtags'] for h in hashtags]
#         hashtag_diversity = len(set(all_hashtags)) / max(len(all_hashtags), 1)
#     else:
#         hashtag_diversity = 0
    
#     # Overall risk score
#     risk_score = (
#         0.3 * min(avg_suspicion, 1.0) +
#         0.2 * min(posting_frequency / 100, 1.0) +
#         0.3 * min(high_conf_posts / 10, 1.0) +
#         0.1 * min(burstiness / 10, 1.0) +
#         0.1 * (1 - hashtag_diversity)
#     )
    
#     user_risk_scores[user_id] = {
#         'posting_frequency': posting_frequency,
#         'avg_suspicion': avg_suspicion,
#         'high_conf_posts': high_conf_posts,
#         'burstiness': burstiness,
#         'hashtag_diversity': hashtag_diversity,
#         'overall_risk': risk_score
#     }

# # Top risky accounts
# top_risky_accounts = sorted(user_risk_scores.items(), key=lambda x: x[1]['overall_risk'], reverse=True)[:10]

# print(f"\n📊 Top Risky Accounts:")
# for user, metrics in top_risky_accounts:
#     print(f"   {user}: Risk={metrics['overall_risk']:.4f}, Posts={metrics['posting_frequency']}, Suspicion={metrics['avg_suspicion']:.4f}")

# # ============================================================================
# # 4. CAMPAIGN DETECTION - COORDINATION PATTERNS
# # ============================================================================
# print("\n[4] Detecting Coordination Campaigns...")

# # Find temporal clusters of suspicious posts
# suspicious_df = merged[merged['is_suspicious'] == True].copy()

# if 'timestamp' in suspicious_df.columns:
#     suspicious_df['timestamp'] = pd.to_datetime(suspicious_df['timestamp'])
#     suspicious_df = suspicious_df.sort_values('timestamp')
    
#     # Detect bursts (posts within 5-minute windows)
#     campaigns = []
#     window = pd.Timedelta(minutes=5)
    
#     for i in range(len(suspicious_df)):
#         post = suspicious_df.iloc[i]
#         time = post['timestamp']
        
#         # Find posts in time window
#         window_posts = suspicious_df[
#             (suspicious_df['timestamp'] >= time) & 
#             (suspicious_df['timestamp'] <= time + window)
#         ]
        
#         if len(window_posts) >= 5:  # At least 5 posts in burst
#             # Check for content similarity
#             unique_users = window_posts['username'].nunique()
            
#             if unique_users >= 3:  # At least 3 different users
#                 campaigns.append({
#                     'start_time': time,
#                     'num_posts': len(window_posts),
#                     'num_users': unique_users,
#                     'avg_suspicion': window_posts['suspicion_score'].mean(),
#                     'post_ids': window_posts['post_id'].tolist()[:10]
#                 })
    
#     # Remove duplicates
#     unique_campaigns = []
#     seen_times = set()
#     for camp in campaigns:
#         time_key = camp['start_time'].strftime('%Y-%m-%d %H:%M')
#         if time_key not in seen_times:
#             unique_campaigns.append(camp)
#             seen_times.add(time_key)
    
#     print(f"\n📊 Detected {len(unique_campaigns)} Coordination Campaigns:")
#     for i, camp in enumerate(unique_campaigns[:5], 1):
#         print(f"   Campaign {i}:")
#         print(f"      Time: {camp['start_time']}")
#         print(f"      Posts: {camp['num_posts']}, Users: {camp['num_users']}")
#         print(f"      Avg Suspicion: {camp['avg_suspicion']:.4f}")
# else:
#     unique_campaigns = []
#     print("   ⚠️ No timestamp data available for campaign detection")

# # ============================================================================
# # 5. INTERACTIVE VISUALIZATIONS
# # ============================================================================
# print("\n[5] Creating Interactive Visualizations...")

# output_dir = Path("explainability_results")
# output_dir.mkdir(exist_ok=True)

# # 5.1: Text Attribution Heatmap
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # Word cloud of suspicious phrases
# ax = axes[0, 0]
# if top_phrases:
#     phrases = [p[0] for p in top_phrases[:20]]
#     scores = [p[1] for p in top_phrases[:20]]
    
#     colors = plt.cm.Reds(np.array(scores) / max(scores))
#     ax.barh(range(len(phrases)), scores, color=colors, edgecolor='black')
#     ax.set_yticks(range(len(phrases)))
#     ax.set_yticklabels(phrases, fontsize=9)
#     ax.set_xlabel('Attribution Score')
#     ax.set_title('Top Suspicious Phrases (Text Attribution)', fontweight='bold')
#     ax.invert_yaxis()
#     ax.grid(axis='x', alpha=0.3)

# # 5.2: User Risk Distribution
# ax = axes[0, 1]
# risk_scores = [metrics['overall_risk'] for metrics in user_risk_scores.values()]
# ax.hist(risk_scores, bins=30, color='orange', edgecolor='black', alpha=0.7)
# ax.set_xlabel('Risk Score')
# ax.set_ylabel('Number of Users')
# ax.set_title('User Risk Score Distribution', fontweight='bold')
# ax.axvline(np.percentile(risk_scores, 90), color='red', linestyle='--', 
#           label='90th Percentile', linewidth=2)
# ax.legend()
# ax.grid(axis='y', alpha=0.3)

# # 5.3: Campaign Timeline
# ax = axes[1, 0]
# if unique_campaigns:
#     campaign_times = [camp['start_time'] for camp in unique_campaigns]
#     campaign_sizes = [camp['num_posts'] for camp in unique_campaigns]
    
#     ax.scatter(campaign_times, campaign_sizes, s=100, c='crimson', 
#               edgecolor='black', alpha=0.7, zorder=3)
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Posts in Burst')
#     ax.set_title('Detected Coordination Campaigns', fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
# else:
#     ax.text(0.5, 0.5, 'No campaigns detected', ha='center', va='center', 
#            fontsize=12, transform=ax.transAxes)
#     ax.set_title('Detected Coordination Campaigns', fontweight='bold')

# # 5.4: Detection Method Contribution
# ax = axes[1, 1]
# method_names = ['Isolation\nForest', 'DBSCAN\nOutliers', 'High\nDistance', 'Deception\nCluster']
# method_contributions = [
#     high_conf['iso_forest_flag'].mean(),
#     high_conf['dbscan_outlier'].mean(),
#     high_conf['high_distance'].mean(),
#     high_conf['in_deception_cluster'].mean()
# ]

# colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
# ax.bar(method_names, method_contributions, color=colors, edgecolor='black', alpha=0.8)
# ax.set_ylabel('Detection Rate')
# ax.set_title('Detection Method Contributions\n(High-Confidence Posts)', fontweight='bold')
# ax.set_ylim([0, 1])
# ax.grid(axis='y', alpha=0.3)

# # Add percentage labels
# for i, v in enumerate(method_contributions):
#     ax.text(i, v + 0.03, f'{v*100:.1f}%', ha='center', fontweight='bold')

# plt.tight_layout()
# plt.savefig(output_dir / 'explainability_dashboard.png', dpi=150, bbox_inches='tight')
# print(f"✅ Saved explainability dashboard")

# # ============================================================================
# # 6. GENERATE INDIVIDUAL POST EXPLANATIONS
# # ============================================================================
# print("\n[6] Generating Individual Post Explanations...")

# def generate_explanation(post_row, word_scores, user_risk_scores):
#     """Generate human-readable explanation for a suspicious post"""
    
#     explanation = {
#         'post_id': post_row['post_id'],
#         'suspicion_score': post_row['suspicion_score'],
#         'detection_methods': [],
#         'text_highlights': [],
#         'user_risk': None,
#         'campaign_involvement': None,
#         'overall_summary': ''
#     }
    
#     # Detection methods
#     if post_row.get('iso_forest_flag', 0):
#         explanation['detection_methods'].append("Anomalous embedding pattern (Isolation Forest)")
#     if post_row.get('dbscan_outlier', 0):
#         explanation['detection_methods'].append("Outlier in behavioral clusters (DBSCAN)")
#     if post_row.get('high_distance', 0):
#         explanation['detection_methods'].append("Isolated from normal content (High distance)")
#     if post_row.get('in_deception_cluster', 0):
#         explanation['detection_methods'].append("Member of deception cluster")
    
#     # Text highlights
#     text = str(post_row.get('post_text', post_row.get('text', '')))
#     words = text.lower().split()
    
#     highlighted_words = []
#     for word in words:
#         if word in word_scores and word_scores[word] > 0.3:
#             highlighted_words.append((word, word_scores[word]))
    
#     explanation['text_highlights'] = sorted(highlighted_words, key=lambda x: x[1], reverse=True)[:5]
    
#     # User risk
#     username = post_row.get('username')
#     if username in user_risk_scores:
#         user_risk = user_risk_scores[username]
#         explanation['user_risk'] = {
#             'overall_score': user_risk['overall_risk'],
#             'posting_frequency': user_risk['posting_frequency'],
#             'avg_suspicion': user_risk['avg_suspicion']
#         }
    
#     # Overall summary
#     num_methods = len(explanation['detection_methods'])
#     suspicion = explanation['suspicion_score']
    
#     if num_methods >= 3 and suspicion > 0.75:
#         confidence = "Very High"
#     elif num_methods >= 2 and suspicion > 0.5:
#         confidence = "High"
#     elif num_methods >= 1 and suspicion > 0.25:
#         confidence = "Medium"
#     else:
#         confidence = "Low"
    
#     explanation['overall_summary'] = (
#         f"This post has a {confidence.lower()} confidence suspicion score of {suspicion:.2f}. "
#         f"It was flagged by {num_methods} detection method(s). "
#     )
    
#     if explanation['text_highlights']:
#         top_word = explanation['text_highlights'][0][0]
#         explanation['overall_summary'] += f"Key suspicious phrase: '{top_word}'. "
    
#     if explanation['user_risk']:
#         risk = explanation['user_risk']['overall_score']
#         if risk > 0.7:
#             explanation['overall_summary'] += "The posting account shows high-risk behavior patterns."
    
#     return explanation

# # Generate explanations for top 20 high-confidence posts
# top_suspicious = high_conf.nlargest(20, 'suspicion_score')
# explanations = []

# for idx, row in top_suspicious.iterrows():
#     exp = generate_explanation(row, word_scores, user_risk_scores)
#     explanations.append(exp)

# # Save explanations
# explanations_df = pd.DataFrame([
#     {
#         'post_id': exp['post_id'],
#         'suspicion_score': exp['suspicion_score'],
#         'num_detection_methods': len(exp['detection_methods']),
#         'detection_methods': '; '.join(exp['detection_methods']),
#         'top_suspicious_words': ', '.join([w for w, s in exp['text_highlights']]),
#         'user_risk_score': exp['user_risk']['overall_score'] if exp['user_risk'] else None,
#         'explanation_summary': exp['overall_summary']
#     }
#     for exp in explanations
# ])

# explanations_df.to_csv(output_dir / 'post_explanations.csv', index=False)
# print(f"✅ Saved explanations for {len(explanations)} posts")

# # ============================================================================
# # 7. SAVE ALL ARTIFACTS
# # ============================================================================
# print("\n[7] Saving Explainability Artifacts...")

# # Save word attribution scores
# pd.DataFrame(top_phrases, columns=['phrase', 'score']).to_csv(
#     output_dir / 'suspicious_phrases.csv', index=False
# )

# # Save user risk scores
# pd.DataFrame.from_dict(user_risk_scores, orient='index').to_csv(
#     output_dir / 'user_risk_scores.csv'
# )

# # Save campaigns
# if unique_campaigns:
#     pd.DataFrame(unique_campaigns).to_csv(
#         output_dir / 'detected_campaigns.csv', index=False
#     )

# print(f"✅ Saved all artifacts to {output_dir}/")

# # ============================================================================
# # SUMMARY REPORT
# # ============================================================================
# print("\n" + "="*80)
# print("✅ EXPLAINABILITY LAYER COMPLETE!")
# print("="*80)

# print(f"\n📊 Summary Statistics:")
# print(f"   Analyzed Posts: {len(merged)}")
# print(f"   High-Confidence Suspicious: {len(high_conf)}")
# print(f"   Suspicious Phrases Identified: {len(top_phrases)}")
# print(f"   Risky Accounts: {len([u for u, m in user_risk_scores.items() if m['overall_risk'] > 0.5])}")
# print(f"   Coordination Campaigns: {len(unique_campaigns) if unique_campaigns else 0}")

# print(f"\n📁 Explainability Results:")
# print(f"   {output_dir}/")
# print(f"   ├── explainability_dashboard.png      - Main visualization")
# print(f"   ├── post_explanations.csv             - Individual post explanations")
# print(f"   ├── suspicious_phrases.csv            - Text attribution scores")
# print(f"   ├── user_risk_scores.csv              - Account risk indicators")
# print(f"   └── detected_campaigns.csv            - Coordination campaigns")

# print(f"\n💡 Key Insights:")
# if top_phrases:
#     print(f"   Top suspicious phrase: '{top_phrases[0][0]}' (score: {top_phrases[0][1]:.4f})")
# if top_risky_accounts:
#     print(f"   Riskiest account: {top_risky_accounts[0][0]} (risk: {top_risky_accounts[0][1]['overall_risk']:.4f})")
# if unique_campaigns:
#     largest_campaign = max(unique_campaigns, key=lambda x: x['num_posts'])
#     print(f"   Largest campaign: {largest_campaign['num_posts']} posts by {largest_campaign['num_users']} users")

# print("\n🎯 Use post_explanations.csv for analyst review!")

# import pandas as pd

# # Path to your CSV file
# file_path = "explainability_results/post_explanations.csv"

# # Load the CSV into a pandas DataFrame
# post_explanations = pd.read_csv(file_path)

# # Check the first few rows
# print(post_explanations.head())


"""
EXPLAINABILITY LAYER FOR SUSPICIOUS CONTENT DETECTION
======================================================
Provides interpretable explanations for why posts are flagged as suspicious.

Components:
1. Text Attribution - Highlights suspicious words/phrases
2. Graph Explanation - Shows influence and spread patterns
3. Metadata Analysis - Account-level risk indicators
4. Campaign Detection - Coordination patterns
5. Interactive Visualizations - Analyst-friendly dashboards
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("EXPLAINABILITY LAYER FOR SUSPICIOUS CONTENT DETECTION")
print("="*80)

# ============================================================================
# LOAD DETECTION RESULTS & DATA
# ============================================================================
print("\n📦 Loading detection results and data...")

# Load detection results
detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
high_conf = pd.read_csv("suspicious_detection_results/high_confidence_suspicious.csv")

# Load original data
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(int)

# Merge
merged = detections.merge(df, on='post_id', how='left')

print(f"✅ Loaded {len(detections)} detections")
print(f"✅ Loaded {len(high_conf)} high-confidence detections")

# ============================================================================
# LOAD HETEROGENEOUS GRAPH
# ============================================================================
print("\n📦 Loading pre-trained Temporal GNN model and graph...")
node_features, edge_dict, node_mappings = load_heterogeneous_graph()

# Normalize features
for ntype, features in node_features.items():
    features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    node_features[ntype] = F.normalize(features, p=2, dim=1).to(device)

edge_dict = {k: (ei.to(device), ew.to(device)) for k, (ei, ew) in edge_dict.items()}

# Prepare node_dims for TemporalHeterogeneousGNN
node_dims = {ntype: features.shape[1] for ntype, features in node_features.items()}

# Initialize GNN model
gnn_model = TemporalHeterogeneousGNN(node_dims=node_dims)
gnn_model.to(device).eval()

# ============================================================================
# 1. TEXT ATTRIBUTION - HIGHLIGHT SUSPICIOUS PHRASES
# ============================================================================
print("\n[1] Computing Text Attribution Scores...")

def compute_text_attribution(text_series, suspicion_scores):
    """
    Compute word importance scores based on correlation with suspicion.
    Returns dict of {word: importance_score}
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.stats import spearmanr
    
    texts = text_series.fillna('').astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    word_scores = {}
    for i, word in enumerate(feature_names):
        word_tfidf = tfidf_matrix[:, i].toarray().flatten()
        corr, _ = spearmanr(word_tfidf, suspicion_scores)
        if not np.isnan(corr):
            word_scores[word] = abs(corr)
    
    return word_scores, vectorizer

text_col = 'post_text' if 'post_text' in merged.columns else 'text'
word_scores, vectorizer = compute_text_attribution(
    merged[text_col], 
    merged['suspicion_score']
)

top_phrases = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:30]

print(f"\n📊 Top Suspicious Phrases:")
for phrase, score in top_phrases[:10]:
    print(f"   '{phrase}': {score:.4f}")

# ============================================================================
# 2. GRAPH EXPLANATION - INFLUENCE & SPREAD PATTERNS
# ============================================================================
print("\n[2] Analyzing Graph Influence Patterns...")

import pickle
with open("heterogeneous_graph/edges.pkl", "rb") as f:
    edges = pickle.load(f)

suspicious_posts = set(high_conf['post_id'].values)
user_post_edges = edges.get(('user', 'creates', 'post'), [])

influence_graph = nx.DiGraph()

for src, dst, weight, timestamp in user_post_edges:
    user_id = [uid for uid, idx in node_mappings['user'].items() if idx == src]
    post_id = [pid for pid, idx in node_mappings['post'].items() if idx == dst]
    
    if user_id and post_id:
        user_id = user_id[0]
        post_id = post_id[0]
        if post_id in suspicious_posts:
            influence_graph.add_edge(user_id, post_id, weight=weight, timestamp=timestamp)

print(f"✅ Built influence graph: {len(influence_graph.nodes)} nodes, {len(influence_graph.edges)} edges")

if len(influence_graph.nodes) > 0:
    try:
        pagerank = nx.pagerank(influence_graph)
        top_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n📊 Top Influential Users in Suspicious Network:")
        for user, score in top_users:
            print(f"   User {user}: PageRank={score:.4f}")
    except:
        print("   ⚠️ Could not compute centrality (graph too small)")

# ============================================================================
# 3. METADATA ANALYSIS - ACCOUNT RISK INDICATORS
# ============================================================================
print("\n[3] Computing Account Risk Indicators...")

user_risk_scores = {}

for user_id in merged['username'].unique():
    user_posts = merged[merged['username'] == user_id]
    if len(user_posts) == 0: continue
    posting_frequency = len(user_posts)
    avg_suspicion = user_posts['suspicion_score'].mean()
    high_conf_posts = (user_posts['suspicion_score'] > 0.5).sum()
    
    if 'timestamp' in user_posts.columns:
        times = pd.to_datetime(
        user_posts['timestamp'].astype(str).str.replace(r'\s+:\s+', ':', regex=True),
        errors='coerce'
        ).dropna().sort_values()
        if len(times) > 1:
            time_diffs = times.diff().dt.total_seconds().dropna()
            burstiness = time_diffs.std() / (time_diffs.mean() + 1e-10) if len(time_diffs) > 0 else 0
        else:
            burstiness = 0
    else:
        burstiness = 0
    
    if 'hashtags' in user_posts.columns:
        all_hashtags = [h for hashtags in user_posts['hashtags'] for h in hashtags]
        hashtag_diversity = len(set(all_hashtags)) / max(len(all_hashtags), 1)
    else:
        hashtag_diversity = 0
    
    risk_score = (
        0.3 * min(avg_suspicion, 1.0) +
        0.2 * min(posting_frequency / 100, 1.0) +
        0.3 * min(high_conf_posts / 10, 1.0) +
        0.1 * min(burstiness / 10, 1.0) +
        0.1 * (1 - hashtag_diversity)
    )
    
    user_risk_scores[user_id] = {
        'posting_frequency': posting_frequency,
        'avg_suspicion': avg_suspicion,
        'high_conf_posts': high_conf_posts,
        'burstiness': burstiness,
        'hashtag_diversity': hashtag_diversity,
        'overall_risk': risk_score
    }

top_risky_accounts = sorted(user_risk_scores.items(), key=lambda x: x[1]['overall_risk'], reverse=True)[:10]

print(f"\n📊 Top Risky Accounts:")
for user, metrics in top_risky_accounts:
    print(f"   {user}: Risk={metrics['overall_risk']:.4f}, Posts={metrics['posting_frequency']}, Suspicion={metrics['avg_suspicion']:.4f}")

# ============================================================================
# 4. CAMPAIGN DETECTION - COORDINATION PATTERNS
# ============================================================================
print("\n[4] Detecting Coordination Campaigns...")

suspicious_df = merged[merged['is_suspicious'] == True].copy()

if 'timestamp' in suspicious_df.columns:
    suspicious_df['timestamp'] = pd.to_datetime(suspicious_df['timestamp'])
    suspicious_df = suspicious_df.sort_values('timestamp')
    
    campaigns = []
    window = pd.Timedelta(minutes=5)
    
    for i in range(len(suspicious_df)):
        post = suspicious_df.iloc[i]
        time = post['timestamp']
        window_posts = suspicious_df[
            (suspicious_df['timestamp'] >= time) & 
            (suspicious_df['timestamp'] <= time + window)
        ]
        if len(window_posts) >= 5:
            unique_users = window_posts['username'].nunique()
            if unique_users >= 3:
                campaigns.append({
                    'start_time': time,
                    'num_posts': len(window_posts),
                    'num_users': unique_users,
                    'avg_suspicion': window_posts['suspicion_score'].mean(),
                    'post_ids': window_posts['post_id'].tolist()[:10]
                })
    
    unique_campaigns = []
    seen_times = set()
    for camp in campaigns:
        time_key = camp['start_time'].strftime('%Y-%m-%d %H:%M')
        if time_key not in seen_times:
            unique_campaigns.append(camp)
            seen_times.add(time_key)
    
    print(f"\n📊 Detected {len(unique_campaigns)} Coordination Campaigns:")
    for i, camp in enumerate(unique_campaigns[:5], 1):
        print(f"   Campaign {i}:")
        print(f"      Time: {camp['start_time']}")
        print(f"      Posts: {camp['num_posts']}, Users: {camp['num_users']}")
        print(f"      Avg Suspicion: {camp['avg_suspicion']:.4f}")
else:
    unique_campaigns = []
    print("   ⚠️ No timestamp data available for campaign detection")

# ============================================================================
# 5. INTERACTIVE VISUALIZATIONS
# ============================================================================
print("\n[5] Creating Interactive Visualizations...")
output_dir = Path("explainability_results")
output_dir.mkdir(exist_ok=True)

# (The visualization code remains unchanged; omitted for brevity here but is the same as your script.)

# ============================================================================
# 6. GENERATE INDIVIDUAL POST EXPLANATIONS
# ============================================================================
print("\n[6] Generating Individual Post Explanations...")

def generate_explanation(post_row, word_scores, user_risk_scores):
    """Generate human-readable explanation for a suspicious post"""
    explanation = {
        'post_id': post_row['post_id'],
        'suspicion_score': post_row['suspicion_score'],
        'detection_methods': [],
        'text_highlights': [],
        'user_risk': None,
        'campaign_involvement': None,
        'overall_summary': ''
    }
    
    if post_row.get('iso_forest_flag', 0):
        explanation['detection_methods'].append("Anomalous embedding pattern (Isolation Forest)")
    if post_row.get('dbscan_outlier', 0):
        explanation['detection_methods'].append("Outlier in behavioral clusters (DBSCAN)")
    if post_row.get('high_distance', 0):
        explanation['detection_methods'].append("Isolated from normal content (High distance)")
    if post_row.get('in_deception_cluster', 0):
        explanation['detection_methods'].append("Member of deception cluster")
    
    text = str(post_row.get('post_text', post_row.get('text', '')))
    words = text.lower().split()
    
    highlighted_words = []
    for word in words:
        if word in word_scores and word_scores[word] > 0.3:
            highlighted_words.append((word, word_scores[word]))
    explanation['text_highlights'] = sorted(highlighted_words, key=lambda x: x[1], reverse=True)[:5]
    
    username = post_row.get('username')
    if username in user_risk_scores:
        user_risk = user_risk_scores[username]
        explanation['user_risk'] = {
            'overall_score': user_risk['overall_risk'],
            'posting_frequency': user_risk['posting_frequency'],
            'avg_suspicion': user_risk['avg_suspicion']
        }
    
    num_methods = len(explanation['detection_methods'])
    suspicion = explanation['suspicion_score']
    if num_methods >= 3 and suspicion > 0.75:
        confidence = "Very High"
    elif num_methods >= 2 and suspicion > 0.5:
        confidence = "High"
    elif num_methods >= 1 and suspicion > 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    explanation['overall_summary'] = (
        f"This post has a {confidence.lower()} confidence suspicion score of {suspicion:.2f}. "
        f"It was flagged by {num_methods} detection method(s). "
    )
    
    if explanation['text_highlights']:
        top_word = explanation['text_highlights'][0][0]
        explanation['overall_summary'] += f"Key suspicious phrase: '{top_word}'. "
    
    if explanation['user_risk']:
        risk = explanation['user_risk']['overall_score']
        if risk > 0.7:
            explanation['overall_summary'] += "The posting account shows high-risk behavior patterns."
    
    return explanation

top_suspicious = high_conf.nlargest(20, 'suspicion_score')
explanations = []

for idx, row in top_suspicious.iterrows():
    exp = generate_explanation(row, word_scores, user_risk_scores)
    explanations.append(exp)

explanations_df = pd.DataFrame([
    {
        'post_id': exp['post_id'],
        'suspicion_score': exp['suspicion_score'],
        'num_detection_methods': len(exp['detection_methods']),
        'detection_methods': '; '.join(exp['detection_methods']),
        'top_suspicious_words': ', '.join([w for w, s in exp['text_highlights']]),
        'user_risk_score': exp['user_risk']['overall_score'] if exp['user_risk'] else None,
        'explanation_summary': exp['overall_summary']
    }
    for exp in explanations
])

explanations_df.to_csv(output_dir / 'post_explanations.csv', index=False)
print(f"✅ Saved explanations for {len(explanations)} posts")

# ============================================================================
# 7. SAVE ALL ARTIFACTS
# ============================================================================
print("\n[7] Saving Explainability Artifacts...")

pd.DataFrame(top_phrases, columns=['phrase', 'score']).to_csv(output_dir / 'suspicious_phrases.csv', index=False)
pd.DataFrame.from_dict(user_risk_scores, orient='index').to_csv(output_dir / 'user_risk_scores.csv')
if unique_campaigns:
    pd.DataFrame(unique_campaigns).to_csv(output_dir / 'detected_campaigns.csv', index=False)

print(f"✅ Saved all artifacts to {output_dir}/")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("✅ EXPLAINABILITY LAYER COMPLETE!")
print("="*80)

"""
EXPLAINABILITY LAYER WITH FUSION WEIGHTS
=========================================
Enhanced explainability that includes modality fusion weights
Shows which modality (text, image, meta) contributed most to each detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import networkx as nx
from datetime import datetime
import warnings
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings('ignore')

from temporal_graph import TemporalHeterogeneousGNN, load_heterogeneous_graph
from rough_work import EmotionAwareFakeNewsDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("EXPLAINABILITY LAYER WITH FUSION WEIGHTS")
print("="*80)

# ============================================================================
# STEP 1: EXTRACT FUSION WEIGHTS FROM TRAINED MODEL
# ============================================================================
print("\n[STEP 1] Extracting fusion weights from trained model...")

# Load dataframe
df = pd.read_pickle("Dataset/twitter/df_preprocessed_with_scores.pkl")
df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
df['post_id'] = df['post_id'].astype(int)

# Load embeddings
print("Loading embeddings...")
text_embeddings = torch.tensor(np.array(df["semantic_vector"].tolist()), dtype=torch.float32)

with open("Dataset/twitter/image_embeddings_cache.pkl", "rb") as f:
    cache = pickle.load(f)
image_embeddings = cache["image_embeddings"]

metadata_embeddings = torch.load("metadata_user_sequence_embeddings.pt")
if metadata_embeddings.dim() == 3:
    metadata_embeddings = metadata_embeddings.squeeze(1)

vad_data = torch.load("Dataset/twitter/prepared_vad_data.pt")

# Align sizes
N = len(df)
for emb in [image_embeddings, metadata_embeddings]:
    if len(emb) > N:
        emb = emb[:N]
    elif len(emb) < N:
        pad = torch.zeros(N - len(emb), emb.shape[1])
        emb = torch.cat([emb, pad], dim=0)

for k in ["vad_text", "vad_image", "affective_meta"]:
    if len(vad_data[k]) > N:
        vad_data[k] = vad_data[k][:N]
    elif len(vad_data[k]) < N:
        pad = torch.zeros(N - len(vad_data[k]), vad_data[k].shape[1])
        vad_data[k] = torch.cat([vad_data[k], pad], dim=0)

# Load trained model
print("Loading trained emotion-aware model...")
emotion_model = EmotionAwareFakeNewsDetector(
    d_text=128,
    d_image=1024,
    d_meta=128,
    d_common=256,
    vad_dim=3,
    meta_affective_dim=128,
    mismatch_dim=128,
    temporal_hidden=64,
    num_classes=1
).to(device)

state = torch.load("checkpoints/best_emotion_aware_detector.pth", map_location=device)

# Remap keys
new_state = {}
for k, v in state.items():
    if k.startswith("fusion."):
        new_k = k.replace("fusion.", "fusion_layer.")
        new_state[new_k] = v
    else:
        if not k.startswith("classifier."):
            new_state[k] = v

emotion_model.load_state_dict(new_state, strict=False)
emotion_model.eval()

# Create dataset
class FusionWeightDataset(Dataset):
    def __init__(self, df, text_emb, image_emb, meta_emb, vad_data):
        self.df = df
        self.text = text_emb
        self.image = image_emb
        self.meta = meta_emb
        self.vad = vad_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "image": self.image[idx],
            "meta": self.meta[idx],
            "vad_text": self.vad["vad_text"][idx],
            "vad_image": self.vad["vad_image"][idx],
            "affective_meta": self.vad["affective_meta"][idx],
            "post_id": self.df.iloc[idx]["post_id"],
        }

dataset = FusionWeightDataset(df, text_embeddings, image_embeddings, metadata_embeddings, vad_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Extract fusion weights
print("Extracting fusion weights for all posts...")
all_fusion_weights = []
all_post_ids = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing batches"):
        h_text = batch['text'].to(device)
        h_image = batch['image'].to(device)
        h_meta = batch['meta'].to(device)
        vad_text = batch['vad_text'].to(device)
        vad_image = batch['vad_image'].to(device)
        affective_meta = batch['affective_meta'].to(device)

        logits, intermediates = emotion_model(
            h_text, h_image, h_meta,
            affective_meta=affective_meta,
            vad_text=vad_text,
            vad_image=vad_image
        )
        
        # Extract emotion_weights (fusion weights)
        emotion_weights = intermediates['emotion_weights']  # Shape: [batch, 3]
        
        all_fusion_weights.append(emotion_weights.cpu())
        all_post_ids.extend(batch['post_id'].tolist())

# Concatenate all weights
fusion_weights_tensor = torch.cat(all_fusion_weights, dim=0)  # Shape: [N, 3]

# Create fusion weights dataframe
fusion_df = pd.DataFrame({
    'post_id': all_post_ids,
    'text_weight': fusion_weights_tensor[:, 0].numpy(),
    'image_weight': fusion_weights_tensor[:, 1].numpy(),
    'meta_weight': fusion_weights_tensor[:, 2].numpy()
})

# Add dominant modality
modality_names = ['text', 'image', 'meta']
fusion_df['dominant_modality'] = [
    modality_names[idx] for idx in fusion_weights_tensor.argmax(dim=1).numpy()
]

print(f"✅ Extracted fusion weights for {len(fusion_df)} posts")
print(f"\nSample fusion weights:")
print(fusion_df.head(10))

# ============================================================================
# STEP 2: LOAD DETECTION RESULTS & MERGE WITH FUSION WEIGHTS
# ============================================================================
print("\n[STEP 2] Loading detection results and merging with fusion weights...")

detections = pd.read_csv("suspicious_detection_results/suspicious_posts_detected.csv")
high_conf = pd.read_csv("suspicious_detection_results/high_confidence_suspicious.csv")

# Merge with fusion weights
detections = detections.merge(fusion_df, on='post_id', how='left')
high_conf = high_conf.merge(fusion_df, on='post_id', how='left')

# Merge with original data
merged = detections.merge(df, on='post_id', how='left')

print(f"✅ Merged {len(detections)} detections with fusion weights")

# ============================================================================
# STEP 3: TEXT ATTRIBUTION
# ============================================================================
print("\n[STEP 3] Computing Text Attribution Scores...")

def compute_text_attribution(text_series, suspicion_scores):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.stats import spearmanr
    
    texts = text_series.fillna('').astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    word_scores = {}
    for i, word in enumerate(feature_names):
        word_tfidf = tfidf_matrix[:, i].toarray().flatten()
        corr, _ = spearmanr(word_tfidf, suspicion_scores)
        if not np.isnan(corr):
            word_scores[word] = abs(corr)
    
    return word_scores, vectorizer

text_col = 'post_text' if 'post_text' in merged.columns else 'text'
word_scores, vectorizer = compute_text_attribution(
    merged[text_col], 
    merged['suspicion_score']
)

top_phrases = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:30]

print(f"\n📊 Top Suspicious Phrases:")
for phrase, score in top_phrases[:10]:
    print(f"   '{phrase}': {score:.4f}")

# ============================================================================
# STEP 4: ANALYZE FUSION WEIGHTS BY DETECTION CATEGORY
# ============================================================================
print("\n[STEP 4] Analyzing fusion weights by detection category...")

# Analyze which modality dominates suspicious vs normal posts
suspicious_fusion = merged[merged['is_suspicious'] == True][['text_weight', 'image_weight', 'meta_weight']]
normal_fusion = merged[merged['is_suspicious'] == False][['text_weight', 'image_weight', 'meta_weight']]

print(f"\n📊 Average Fusion Weights - Suspicious Posts:")
print(f"   Text: {suspicious_fusion['text_weight'].mean():.4f}")
print(f"   Image: {suspicious_fusion['image_weight'].mean():.4f}")
print(f"   Meta: {suspicious_fusion['meta_weight'].mean():.4f}")

print(f"\n📊 Average Fusion Weights - Normal Posts:")
print(f"   Text: {normal_fusion['text_weight'].mean():.4f}")
print(f"   Image: {normal_fusion['image_weight'].mean():.4f}")
print(f"   Meta: {normal_fusion['meta_weight'].mean():.4f}")

# Dominant modality distribution
print(f"\n📊 Dominant Modality Distribution:")
print(merged[merged['is_suspicious'] == True]['dominant_modality'].value_counts())

# ============================================================================
# STEP 5: GENERATE ENHANCED EXPLANATIONS WITH FUSION WEIGHTS
# ============================================================================
print("\n[STEP 5] Generating enhanced explanations...")

def generate_explanation_with_fusion(post_row, word_scores, user_risk_scores=None):
    """Generate explanation including fusion weights"""
    explanation = {
        'post_id': post_row['post_id'],
        'suspicion_score': post_row['suspicion_score'],
        'detection_methods': [],
        'text_highlights': [],
        'fusion_weights': {},
        'dominant_modality': post_row.get('dominant_modality', 'unknown'),
        'user_risk': None,
        'overall_summary': ''
    }
    
    # Detection methods
    if post_row.get('iso_forest_flag', 0):
        explanation['detection_methods'].append("Anomalous embedding pattern")
    if post_row.get('dbscan_outlier', 0):
        explanation['detection_methods'].append("Behavioral outlier")
    if post_row.get('high_distance', 0):
        explanation['detection_methods'].append("Isolated from normal content")
    if post_row.get('in_deception_cluster', 0):
        explanation['detection_methods'].append("Member of deception cluster")
    
    # Fusion weights
    explanation['fusion_weights'] = {
        'text': float(post_row.get('text_weight', 0)),
        'image': float(post_row.get('image_weight', 0)),
        'meta': float(post_row.get('meta_weight', 0))
    }
    
    # Text highlights
    text = str(post_row.get('post_text', post_row.get('text', '')))
    words = text.lower().split()
    highlighted_words = []
    for word in words:
        if word in word_scores and word_scores[word] > 0.3:
            highlighted_words.append((word, word_scores[word]))
    explanation['text_highlights'] = sorted(highlighted_words, key=lambda x: x[1], reverse=True)[:5]
    
    # Build summary
    num_methods = len(explanation['detection_methods'])
    suspicion = explanation['suspicion_score']
    
    if num_methods >= 3 and suspicion > 0.75:
        confidence = "Very High"
    elif num_methods >= 2 and suspicion > 0.5:
        confidence = "High"
    elif num_methods >= 1 and suspicion > 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    dominant_mod = explanation['dominant_modality']
    dominant_weight = explanation['fusion_weights'].get(dominant_mod, 0)
    
    explanation['overall_summary'] = (
        f"This post has a {confidence.lower()} confidence suspicion score of {suspicion:.2f}. "
        f"It was flagged by {num_methods} detection method(s). "
        f"The '{dominant_mod}' modality contributed most ({dominant_weight:.2%}) to this detection. "
    )
    
    if explanation['text_highlights'] and dominant_mod == 'text':
        top_word = explanation['text_highlights'][0][0]
        explanation['overall_summary'] += f"Key suspicious phrase: '{top_word}'. "
    elif dominant_mod == 'image':
        explanation['overall_summary'] += "Visual content shows suspicious patterns. "
    elif dominant_mod == 'meta':
        explanation['overall_summary'] += "User behavior metadata indicates anomalous patterns. "
    
    return explanation

# Generate explanations for top suspicious posts
top_suspicious = high_conf.nlargest(20, 'suspicion_score')
explanations = []

for idx, row in top_suspicious.iterrows():
    exp = generate_explanation_with_fusion(row, word_scores)
    explanations.append(exp)

# Save enhanced explanations
explanations_df = pd.DataFrame([
    {
        'post_id': exp['post_id'],
        'suspicion_score': exp['suspicion_score'],
        'num_detection_methods': len(exp['detection_methods']),
        'detection_methods': '; '.join(exp['detection_methods']),
        'dominant_modality': exp['dominant_modality'],
        'text_weight': exp['fusion_weights']['text'],
        'image_weight': exp['fusion_weights']['image'],
        'meta_weight': exp['fusion_weights']['meta'],
        'top_suspicious_words': ', '.join([w for w, s in exp['text_highlights']]),
        'explanation_summary': exp['overall_summary']
    }
    for exp in explanations
])

output_dir = Path("explainability_results")
output_dir.mkdir(exist_ok=True)

explanations_df.to_csv(output_dir / 'post_explanations_with_fusion.csv', index=False)
print(f"✅ Saved enhanced explanations for {len(explanations)} posts")

# ============================================================================
# STEP 6: VISUALIZATIONS WITH FUSION WEIGHTS
# ============================================================================
print("\n[STEP 6] Creating visualizations with fusion weights...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1: Fusion weights distribution
ax = axes[0, 0]
fusion_weights_data = merged[['text_weight', 'image_weight', 'meta_weight']].values
ax.boxplot(fusion_weights_data, labels=['Text', 'Image', 'Meta'])
ax.set_ylabel('Weight Value')
ax.set_title('Fusion Weight Distribution', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 6.2: Dominant modality by suspicion level
ax = axes[0, 1]
high_sus = merged[merged['suspicion_score'] > 0.5]
low_sus = merged[merged['suspicion_score'] <= 0.5]

modality_counts = pd.DataFrame({
    'High Suspicion': high_sus['dominant_modality'].value_counts(),
    'Low Suspicion': low_sus['dominant_modality'].value_counts()
}).fillna(0)

modality_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
ax.set_title('Dominant Modality by Suspicion Level', fontweight='bold')
ax.set_xlabel('Modality')
ax.set_ylabel('Count')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 6.3: Fusion weights heatmap for top suspicious posts
ax = axes[0, 2]
top_20 = high_conf.nlargest(20, 'suspicion_score')
fusion_matrix = top_20[['text_weight', 'image_weight', 'meta_weight']].values
im = ax.imshow(fusion_matrix.T, aspect='auto', cmap='YlOrRd')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Text', 'Image', 'Meta'])
ax.set_xlabel('Post Index (sorted by suspicion)')
ax.set_title('Fusion Weights Heatmap\n(Top 20 Suspicious Posts)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Weight')

# 6.4: Text phrases
ax = axes[1, 0]
if top_phrases:
    phrases = [p[0] for p in top_phrases[:15]]
    scores = [p[1] for p in top_phrases[:15]]
    colors = plt.cm.Reds(np.array(scores) / max(scores))
    ax.barh(range(len(phrases)), scores, color=colors, edgecolor='black')
    ax.set_yticks(range(len(phrases)))
    ax.set_yticklabels(phrases, fontsize=8)
    ax.set_xlabel('Attribution Score')
    ax.set_title('Top Suspicious Phrases', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

# 6.5: Modality importance vs suspicion score
ax = axes[1, 1]
for modality, color in zip(['text_weight', 'image_weight', 'meta_weight'], 
                           ['#3498db', '#e74c3c', '#2ecc71']):
    ax.scatter(merged[modality], merged['suspicion_score'], 
              alpha=0.3, s=20, label=modality.replace('_weight', '').title(), c=color)
ax.set_xlabel('Modality Weight')
ax.set_ylabel('Suspicion Score')
ax.set_title('Modality Weights vs Suspicion Score', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 6.6: Detection method contributions
ax = axes[1, 2]
method_names = ['Isolation\nForest', 'DBSCAN\nOutliers', 'High\nDistance', 'Deception\nCluster']
method_contributions = [
    high_conf['iso_forest_flag'].mean(),
    high_conf['dbscan_outlier'].mean(),
    high_conf['high_distance'].mean(),
    high_conf['in_deception_cluster'].mean()
]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
ax.bar(method_names, method_contributions, color=colors, edgecolor='black', alpha=0.8)
ax.set_ylabel('Detection Rate')
ax.set_title('Detection Method Contributions', fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(method_contributions):
    ax.text(i, v + 0.03, f'{v*100:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'explainability_dashboard_with_fusion.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved enhanced visualization dashboard")

# ============================================================================
# STEP 7: SAVE ALL ARTIFACTS
# ============================================================================
print("\n[STEP 7] Saving all artifacts...")

# Save fusion weights for all posts
fusion_df.to_csv(output_dir / 'fusion_weights_all_posts.csv', index=False)

# Save suspicious phrases
pd.DataFrame(top_phrases, columns=['phrase', 'score']).to_csv(
    output_dir / 'suspicious_phrases.csv', index=False
)

print(f"✅ Saved all artifacts to {output_dir}/")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("✅ ENHANCED EXPLAINABILITY WITH FUSION WEIGHTS COMPLETE!")
print("="*80)

print(f"\n📊 Summary Statistics:")
print(f"   Total Posts Analyzed: {len(merged)}")
print(f"   High-Confidence Suspicious: {len(high_conf)}")
print(f"   Suspicious Phrases Identified: {len(top_phrases)}")

print(f"\n📊 Fusion Weight Insights:")
print(f"   Average Text Weight (Suspicious): {suspicious_fusion['text_weight'].mean():.4f}")
print(f"   Average Image Weight (Suspicious): {suspicious_fusion['image_weight'].mean():.4f}")
print(f"   Average Meta Weight (Suspicious): {suspicious_fusion['meta_weight'].mean():.4f}")

print(f"\n📁 Enhanced Explainability Results:")
print(f"   {output_dir}/")
print(f"   ├── explainability_dashboard_with_fusion.png   - Enhanced visualizations")
print(f"   ├── post_explanations_with_fusion.csv          - Explanations with fusion weights")
print(f"   ├── fusion_weights_all_posts.csv               - All post fusion weights")
print(f"   └── suspicious_phrases.csv                     - Text attribution scores")

print("\n🎯 Fusion weights show which modality (text/image/meta) drove each detection!")
print("🎯 Use post_explanations_with_fusion.csv for detailed analyst review!")