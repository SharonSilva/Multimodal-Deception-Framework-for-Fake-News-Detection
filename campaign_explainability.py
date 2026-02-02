# campaign_explainability.py
# ============================================================
# CAMPAIGN EXPLAINABILITY LAYER
# ============================================================
# Produces:
#   1. Human-readable per-campaign explanations (text + CSV)
#   2. A 12-panel dashboard figure — each plot has a plain-
#      English description underneath explaining WHAT it shows
#      and WHY it matters.
#
# How campaigns are identified (for reference):
#   - The campaign detection pipeline builds a graph where each
#     post is a node.
#   - An edge is drawn between two posts when they are
#     content-similar (cosine sim > 0.5) AND posted close in
#     time (exponential decay with tau=1 hour).
#   - Edges are further weighted by anomaly scores so that
#     suspicious posts pull harder.
#   - Louvain community detection then finds tightly-connected
#     clusters in that graph → those clusters are "campaigns".
#   - A campaign must have ≥ 3 posts to be kept.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CAMPAIGN EXPLAINABILITY LAYER")
print("=" * 80)

# ============================================================
# STEP 1: LOAD CAMPAIGN RESULTS
# ============================================================
print("\n[STEP 1] Loading campaign detection results...")

results_dir = Path("campaign_detection_results")

assignments = pd.read_csv(results_dir / "campaign_assignments.csv")
statistics = pd.read_csv(results_dir / "campaign_statistics.csv")

assignments['timestamp'] = pd.to_datetime(assignments['timestamp'])

print(f"   Posts assigned to campaigns : {len(assignments)}")
print(f"   Total campaigns             : {len(statistics)}")
print(f"   Unique users in campaigns   : {assignments['user_id'].nunique()}")

# ============================================================
# STEP 2: PRE-COMPUTE PER-CAMPAIGN METRICS
# ============================================================
print("\n[STEP 2] Computing per-campaign metrics...")

campaign_metrics = []

for _, row in statistics.iterrows():
    cid = row['campaign_id']
    posts = assignments[assignments['campaign_id'] == cid].copy()

    if len(posts) == 0:
        continue

    # --- temporal ---
    times = posts['timestamp'].sort_values()
    duration_hours = row['time_span_hours']

    # inter-post intervals in minutes
    intervals_min = times.diff().dt.total_seconds().dropna().values / 60.0

    # burst: max posts in any 10-min window
    posts['window'] = times.dt.floor('10min')
    burst_max = posts.groupby('window').size().max()

    # --- user concentration ---
    user_counts = posts['user_id'].value_counts()
    n_users = len(user_counts)
    top_user_posts = int(user_counts.iloc[0])          # most active user
    top_user_share = top_user_posts / len(posts) * 100 # % of posts by top user
    power_users = int((user_counts > 1).sum())         # users with >1 post

    # --- co-posting: pairs of users who posted in the same hour ---
    posts['hour'] = times.dt.floor('H')
    copair_counter = Counter()
    for _, grp in posts.groupby('hour'):
        users_in_hour = grp['user_id'].unique()
        for i, u1 in enumerate(users_in_hour):
            for u2 in users_in_hour[i + 1:]:
                copair_counter[tuple(sorted([u1, u2]))] += 1
    top_copairs = copair_counter.most_common(3)

    # --- threat level (same logic as CampaignInvestigator) ---
    threat_score = 0
    if burst_max > 5:
        threat_score += 3
    elif burst_max > 3:
        threat_score += 1
    if n_users > 0 and power_users / n_users > 0.5:
        threat_score += 3
    elif n_users > 0 and power_users / n_users > 0.2:
        threat_score += 1
    if row['mean_anomaly_score'] > 0.7:
        threat_score += 3
    elif row['mean_anomaly_score'] > 0.5:
        threat_score += 1

    if threat_score >= 7:
        threat_level = "CRITICAL"
    elif threat_score >= 4:
        threat_level = "HIGH"
    elif threat_score >= 2:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    campaign_metrics.append({
        'campaign_id': cid,
        'n_posts': int(row['n_posts']),
        'n_users': n_users,
        'posts_per_user': row['posts_per_user'],
        'duration_hours': duration_hours,
        'coordination_score': row['coordination_score'],
        'mean_anomaly_score': row['mean_anomaly_score'],
        'max_anomaly_score': row['max_anomaly_score'],
        'burst_max': int(burst_max),
        'median_interval_min': float(np.median(intervals_min)) if len(intervals_min) > 0 else 0.0,
        'min_interval_min': float(np.min(intervals_min)) if len(intervals_min) > 0 else 0.0,
        'top_user_posts': top_user_posts,
        'top_user_share_pct': top_user_share,
        'power_users': power_users,
        'top_copairs': top_copairs,
        'threat_level': threat_level,
        'intervals_min': intervals_min,
        'user_counts': user_counts,
        'posts_df': posts,
    })

df_metrics = pd.DataFrame([
    {k: v for k, v in m.items() if k not in ('intervals_min', 'user_counts', 'posts_df', 'top_copairs')}
    for m in campaign_metrics
]).sort_values('coordination_score', ascending=False).reset_index(drop=True)

print(f"   Metrics computed for {len(df_metrics)} campaigns")
print(f"   Threat levels → CRITICAL:{(df_metrics['threat_level']=='CRITICAL').sum()}, "
      f"HIGH:{(df_metrics['threat_level']=='HIGH').sum()}, "
      f"MEDIUM:{(df_metrics['threat_level']=='MEDIUM').sum()}, "
      f"LOW:{(df_metrics['threat_level']=='LOW').sum()}")

# Quick lookup: campaign_id → metrics dict
metrics_by_id = {m['campaign_id']: m for m in campaign_metrics}

# ============================================================
# STEP 3: GENERATE PER-CAMPAIGN TEXT EXPLANATIONS
# ============================================================
print("\n[STEP 3] Generating per-campaign explanations...")

explanations = []

for m in sorted(campaign_metrics, key=lambda x: x['coordination_score'], reverse=True):
    cid = m['campaign_id']

    # --- threat emoji ---
    threat_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "🟡", "LOW": "✅"}[m['threat_level']]

    # --- burst label ---
    if m['burst_max'] > 5:
        burst_label = "🚨 EXTREME burst"
    elif m['burst_max'] > 3:
        burst_label = "⚠️ HIGH burst"
    else:
        burst_label = "🟡 moderate burst"

    # --- posting cadence ---
    if m['median_interval_min'] < 5:
        cadence_label = "🚨 very rapid (< 5 min apart)"
    elif m['median_interval_min'] < 30:
        cadence_label = "⚠️ rapid (< 30 min apart)"
    elif m['median_interval_min'] < 120:
        cadence_label = "🟡 moderate (< 2 hrs apart)"
    else:
        cadence_label = "✅ slow (> 2 hrs apart)"

    # --- user concentration ---
    if m['top_user_share_pct'] > 50:
        conc_label = "🚨 highly concentrated — one user dominates"
    elif m['top_user_share_pct'] > 30:
        conc_label = "⚠️ concentrated — a few users drive most posts"
    else:
        conc_label = "🟡 distributed across many users"

    # --- co-posting summary ---
    if m['top_copairs']:
        copair_lines = []
        for (u1, u2), count in m['top_copairs']:
            copair_lines.append(f"      {u1} ↔ {u2} posted in the same hour {count}x")
        copair_text = "\n".join(copair_lines)
    else:
        copair_text = "      No users posted in the same hour"

    explanation = (
        f"\n{'─' * 70}\n"
        f"  CAMPAIGN #{cid}  |  Threat: {threat_emoji} {m['threat_level']}\n"
        f"{'─' * 70}\n"
        f"  What is this campaign?\n"
        f"    A group of {m['n_posts']} posts by {m['n_users']} user(s) that the system\n"
        f"    detected as coordinated. They share similar content and were posted\n"
        f"    close together in time. The more posts share content AND timing, the\n"
        f"    more likely they are part of an organised effort.\n\n"
        f"  📊 Scale\n"
        f"    Posts: {m['n_posts']}  |  Users: {m['n_users']}  |  "
        f"Posts/User: {m['posts_per_user']:.1f}  |  Duration: {m['duration_hours']:.1f} hrs\n\n"
        f"  ⏱️  Timing\n"
        f"    Burst activity : {burst_label} — up to {m['burst_max']} posts in one 10-min window.\n"
        f"    Posting cadence: {cadence_label}.\n"
        f"    Fastest gap    : {m['min_interval_min']:.1f} min between consecutive posts.\n\n"
        f"  👥 User Concentration\n"
        f"    {conc_label}.\n"
        f"    Top user posted {m['top_user_posts']}x ({m['top_user_share_pct']:.0f}% of campaign).\n"
        f"    Power users (>1 post): {m['power_users']} of {m['n_users']}.\n\n"
        f"  🔗 Co-posting (users active in the same hour)\n"
        f"{copair_text}\n\n"
        f"  📈 Suspicion\n"
        f"    Coordination score : {m['coordination_score']:.2f}\n"
        f"    Mean anomaly score : {m['mean_anomaly_score']:.3f}\n"
        f"    Max anomaly score  : {m['max_anomaly_score']:.3f}\n"
    )

    explanations.append({
        'campaign_id': cid,
        'threat_level': m['threat_level'],
        'explanation': explanation
    })

    print(explanation)

# Save explanations CSV
pd.DataFrame(explanations).to_csv(
    results_dir / "campaign_explanations.csv", index=False
)
print(f"\n✅ Explanations saved to {results_dir}/campaign_explanations.csv")

# ============================================================
# STEP 4: DASHBOARD — 12 panels, each with a description
# ============================================================
print("\n[STEP 4] Generating explainability dashboard...")

# colour palette
C = {
    'red': '#e63946', 'orange': '#f4a261', 'green': '#2a9d8f',
    'blue': '#457b9d', 'purple': '#6a4c93', 'pink': '#e76f51',
    'bg': '#f8f9fa', 'text': '#1d3557'
}

fig = plt.figure(figsize=(24, 28), facecolor='white')
fig.suptitle(
    "Campaign Explainability Dashboard",
    fontsize=22, fontweight='bold', color=C['text'], y=0.98
)

gs = gridspec.GridSpec(
    4, 3,
    hspace=0.45, wspace=0.3,
    left=0.06, right=0.96, top=0.95, bottom=0.02
)

# helper: styled axes
def style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight='bold', color=C['text'], pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=C['text'], labelsize=9)
    ax.xaxis.label.set_color(C['text'])
    ax.yaxis.label.set_color(C['text'])

# helper: add description text below an axes
def add_desc(ax, desc_text):
    ax.text(
        0.5, -0.18, desc_text,
        transform=ax.transAxes, fontsize=8.5, color='#555555',
        ha='center', va='top', style='italic', wrap=True,
        fontfamily='sans-serif'
    )

# ---- data shortcuts ----
cids = df_metrics['campaign_id'].astype(str).values
coord = df_metrics['coordination_score'].values
n_posts = df_metrics['n_posts'].values
n_users = df_metrics['n_users'].values
duration = df_metrics['duration_hours'].values
mean_anom = df_metrics['mean_anomaly_score'].values
burst = df_metrics['burst_max'].values
ppu = df_metrics['posts_per_user'].values
top_share = df_metrics['top_user_share_pct'].values
threat_counts = df_metrics['threat_level'].value_counts()

# ─────────────────────────────────────────────
# 1. THREAT LEVEL DISTRIBUTION
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
colors_threat = [C['red'], C['orange'], C['purple'], C['green']]
vals = [threat_counts.get(t, 0) for t in order]
bars = ax.bar(order, vals, color=colors_threat, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15, str(v),
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=C['text'])
style_ax(ax, "Threat Level Distribution")
ax.set_ylabel("Number of Campaigns")
add_desc(ax,
    "Each campaign is assigned a threat level based on burst activity, user\n"
    "concentration, and anomaly scores. CRITICAL = immediate action needed."
)

# ─────────────────────────────────────────────
# 2. COORDINATION SCORE (top 15)
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
top_n = min(15, len(df_metrics))
top_idx = np.argsort(coord)[-top_n:][::-1]
bar_colors = [C['red'] if coord[i] > np.percentile(coord, 75) else C['orange']
              if coord[i] > np.percentile(coord, 50) else C['blue'] for i in top_idx]
ax.barh(range(top_n), coord[top_idx], color=bar_colors, edgecolor='white', linewidth=1)
ax.set_yticks(range(top_n))
ax.set_yticklabels([f"#{cids[i]}" for i in top_idx], fontsize=8.5)
ax.invert_yaxis()
style_ax(ax, "Top Campaigns by Coordination Score")
ax.set_xlabel("Coordination Score")
add_desc(ax,
    "Coordination score = volume × user-concentration × speed × suspicion.\n"
    "Higher = more signs that posts were organised together."
)

# ─────────────────────────────────────────────
# 3. POSTS vs USERS (coloured by anomaly)
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
sc = ax.scatter(n_users, n_posts, c=mean_anom, cmap='YlOrRd',
                s=80, edgecolors=C['text'], linewidths=0.8, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='Mean Anomaly Score', shrink=0.8)
style_ax(ax, "Posts vs Users per Campaign")
ax.set_xlabel("Unique Users")
ax.set_ylabel("Total Posts")
add_desc(ax,
    "Each dot is one campaign. Colour = how anomalous its posts are on average.\n"
    "Bottom-right = few users posting many times → classic coordinated pattern."
)

# ─────────────────────────────────────────────
# 4. CAMPAIGN DURATION DISTRIBUTION
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
dur_clip = duration[duration < 168]  # < 1 week for readability
ax.hist(dur_clip, bins=min(30, len(dur_clip)), color=C['blue'],
        edgecolor='white', linewidth=1, alpha=0.85)
ax.axvline(np.median(dur_clip), color=C['red'], linestyle='--', linewidth=1.5,
           label=f'Median: {np.median(dur_clip):.1f} hrs')
style_ax(ax, "Campaign Duration (< 1 week)")
ax.set_xlabel("Hours")
ax.set_ylabel("Number of Campaigns")
ax.legend(fontsize=8.5)
add_desc(ax,
    "How long each campaign lasted from first to last post.\n"
    "Short durations with many posts = rapid coordinated push."
)

# ─────────────────────────────────────────────
# 5. BURST ACTIVITY
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
burst_colors = [C['red'] if b > 5 else C['orange'] if b > 3 else C['green'] for b in burst]
ax.bar(range(len(burst)), burst, color=burst_colors, edgecolor='white', linewidth=0.8)
ax.axhline(3, color=C['orange'], linestyle=':', linewidth=1, label='High threshold (3)')
ax.axhline(5, color=C['red'], linestyle=':', linewidth=1, label='Extreme threshold (5)')
style_ax(ax, "Max Burst per Campaign (10-min windows)")
ax.set_xlabel("Campaign Index (sorted by coordination)")
ax.set_ylabel("Posts in Busiest 10 min")
ax.legend(fontsize=8, loc='upper right')
add_desc(ax,
    "The most posts any campaign squeezed into a single 10-minute window.\n"
    "High bursts suggest automated or pre-scheduled posting."
)

# ─────────────────────────────────────────────
# 6. POSTS PER USER DISTRIBUTION
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.hist(ppu, bins=min(25, len(ppu)), color=C['purple'],
        edgecolor='white', linewidth=1, alpha=0.85)
ax.axvline(np.median(ppu), color=C['red'], linestyle='--', linewidth=1.5,
           label=f'Median: {np.median(ppu):.1f}')
style_ax(ax, "Posts per User Across Campaigns")
ax.set_xlabel("Posts / User")
ax.set_ylabel("Number of Campaigns")
ax.legend(fontsize=8.5)
add_desc(ax,
    "How many posts each campaign's users contributed on average.\n"
    "High values = few users doing most of the posting."
)

# ─────────────────────────────────────────────
# 7. TOP-USER SHARE (% of campaign posts)
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
share_colors = [C['red'] if s > 50 else C['orange'] if s > 30 else C['green'] for s in top_share]
ax.bar(range(len(top_share)), top_share, color=share_colors, edgecolor='white', linewidth=0.8)
ax.axhline(50, color=C['red'], linestyle=':', linewidth=1, label='Dominant (50%)')
ax.axhline(30, color=C['orange'], linestyle=':', linewidth=1, label='Concentrated (30%)')
style_ax(ax, "Top User's Share of Each Campaign")
ax.set_xlabel("Campaign Index")
ax.set_ylabel("% of Posts by Single User")
ax.legend(fontsize=8, loc='upper right')
add_desc(ax,
    "What percentage of a campaign's posts came from its most active user.\n"
    "Over 50% from one account is a strong coordination signal."
)

# ─────────────────────────────────────────────
# 8. MEAN ANOMALY SCORE PER CAMPAIGN
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
anom_colors = [C['red'] if a > 0.7 else C['orange'] if a > 0.5 else C['blue'] for a in mean_anom]
sorted_anom_idx = np.argsort(mean_anom)[::-1]
ax.barh(range(len(mean_anom)), mean_anom[sorted_anom_idx],
        color=[anom_colors[i] for i in sorted_anom_idx], edgecolor='white', linewidth=0.8)
ax.set_yticks(range(min(15, len(mean_anom))))
ax.set_yticklabels([f"#{cids[sorted_anom_idx[i]]}" for i in range(min(15, len(mean_anom)))],
                   fontsize=8.5)
style_ax(ax, "Mean Anomaly Score (top 15)")
ax.set_xlabel("Anomaly Score (0–1)")
add_desc(ax,
    "Average anomaly score of all posts inside each campaign.\n"
    "Anomaly score is the ensemble of 4 outlier detectors — higher = more unusual."
)

# ─────────────────────────────────────────────
# 9. DURATION vs COORDINATION (bubble = posts)
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
sc = ax.scatter(duration, coord, s=n_posts * 8, c=mean_anom, cmap='YlOrRd',
                edgecolors=C['text'], linewidths=0.8, alpha=0.8, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='Mean Anomaly', shrink=0.8)
style_ax(ax, "Duration vs Coordination (size = post count)")
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Coordination Score")
add_desc(ax,
    "Short duration + high coordination = rapid organised push.\n"
    "Bubble size = number of posts. Colour = anomaly level."
)

# ─────────────────────────────────────────────
# 10. INTER-POST INTERVAL HEATMAP (top 6 campaigns)
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
top6_ids = df_metrics['campaign_id'].values[:6]
interval_data = []
max_bins = 12
for cid in top6_ids:
    m = metrics_by_id.get(cid)
    if m and len(m['intervals_min']) > 0:
        hist, _ = np.histogram(m['intervals_min'], bins=max_bins, range=(0, 60))
        interval_data.append(hist)
    else:
        interval_data.append(np.zeros(max_bins))

interval_arr = np.array(interval_data, dtype=float)
im = ax.imshow(interval_arr, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(max_bins))
ax.set_xticklabels([f"{i*5}" for i in range(max_bins)], fontsize=7.5)
ax.set_yticks(range(len(top6_ids)))
ax.set_yticklabels([f"#{c}" for c in top6_ids], fontsize=8.5)
plt.colorbar(im, ax=ax, label='Count', shrink=0.8)
style_ax(ax, "Inter-Post Intervals (top 6 campaigns)")
ax.set_xlabel("Minutes Between Posts")
add_desc(ax,
    "How many minutes passed between consecutive posts within each campaign.\n"
    "Bright cells at the left = many posts arriving seconds apart."
)

# ─────────────────────────────────────────────
# 11. USER CONCENTRATION SCATTER
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 1])
power_users_arr = df_metrics['power_users'].values
sc = ax.scatter(n_users, power_users_arr, c=coord, cmap='plasma',
                s=70, edgecolors=C['text'], linewidths=0.8, vmin=0)
plt.colorbar(sc, ax=ax, label='Coordination Score', shrink=0.8)
# annotation line: if ALL users are power users
lim = max(n_users.max(), power_users_arr.max()) + 1
ax.plot([0, lim], [0, lim], color=C['red'], linestyle='--', linewidth=1, alpha=0.5, label='All users = power users')
style_ax(ax, "Total Users vs Power Users (>1 post)")
ax.set_xlabel("Total Users")
ax.set_ylabel("Power Users")
ax.legend(fontsize=8)
add_desc(ax,
    "Power users posted more than once in the same campaign.\n"
    "Points near the diagonal = almost everyone posted multiple times → coordinated."
)

# ─────────────────────────────────────────────
# 12. COMPOSITE RISK SUMMARY TABLE
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 2])
ax.axis('off')

# Build a small table for top 8 campaigns
top8 = df_metrics.head(8)
header = f"{'#':<5} {'Posts':<7} {'Users':<7} {'Burst':<6} {'Anom':<6} {'Threat'}"
rows = [header, "─" * 42]
for _, r in top8.iterrows():
    rows.append(
        f"{int(r['campaign_id']):<5} "
        f"{int(r['n_posts']):<7} "
        f"{int(r['n_users']):<7} "
        f"{int(r['burst_max']):<6} "
        f"{r['mean_anomaly_score']:<6.2f} "
        f"{r['threat_level']}"
    )

table_text = "\n".join(rows)
ax.text(0.05, 0.95, "Composite Risk Summary — Top 8",
        transform=ax.transAxes, fontsize=11, fontweight='bold', color=C['text'],
        va='top')
ax.text(0.05, 0.82, table_text,
        transform=ax.transAxes, fontsize=9, color=C['text'],
        va='top', family='monospace')
ax.text(0.05, 0.08,
        "Each row is one campaign. Burst = max posts in any 10-min window.\n"
        "Anom = mean anomaly score. Threat = overall risk assessment.",
        transform=ax.transAxes, fontsize=8.5, color='#555555', style='italic')

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
output_path = results_dir / "campaign_explainability_dashboard.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Dashboard saved to {output_path}")
print("=" * 80)
print("CAMPAIGN EXPLAINABILITY COMPLETE")
print("=" * 80)