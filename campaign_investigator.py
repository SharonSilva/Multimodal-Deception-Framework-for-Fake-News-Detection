"""
CAMPAIGN INVESTIGATION TOOL
============================
Deep-dive analysis of detected campaigns with:
- Temporal patterns
- User network analysis
- Content similarity analysis
- Narrative evolution tracking
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class CampaignInvestigator:
    """
    Tool for investigating detected disinformation campaigns.
    """
    
    def __init__(self, results_dir="campaign_detection_results"):
        """
        Load campaign detection results.
        
        Args:
            results_dir: Path to campaign detection results
        """
        self.results_dir = Path(results_dir)
        
        print("🔍 Loading campaign data...")
        
        # Load campaign assignments
        self.assignments = pd.read_csv(self.results_dir / "campaign_assignments.csv")
        self.statistics = pd.read_csv(self.results_dir / "campaign_statistics.csv")
        
        # Load graph
        self.graph = nx.read_gpickle(self.results_dir / "post_similarity_graph.gpickle")
        
        print(f"✅ Loaded:")
        print(f"   {len(self.assignments)} post assignments")
        print(f"   {len(self.statistics)} campaigns")
        print(f"   Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_campaign_info(self, campaign_id):
        """
        Get basic info about a campaign.
        
        Args:
            campaign_id: Campaign ID to investigate
            
        Returns:
            Dictionary with campaign information
        """
        stats = self.statistics[self.statistics['campaign_id'] == campaign_id]
        
        if len(stats) == 0:
            return None
        
        stats_dict = stats.iloc[0].to_dict()
        
        # Get posts and users
        campaign_posts = self.assignments[self.assignments['campaign_id'] == campaign_id]
        
        return {
            'stats': stats_dict,
            'posts': campaign_posts
        }
    
    def analyze_temporal_pattern(self, campaign_id):
        """
        Analyze temporal posting pattern of a campaign.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Temporal analysis results
        """
        campaign_posts = self.assignments[self.assignments['campaign_id'] == campaign_id]
        
        if len(campaign_posts) == 0:
            print(f"⚠️  Campaign {campaign_id} not found")
            return None
        
        # Parse timestamps
        timestamps = pd.to_datetime(campaign_posts['timestamp'])
        
        # Time range
        start_time = timestamps.min()
        end_time = timestamps.max()
        duration = end_time - start_time
        
        # Posts per hour
        campaign_posts['hour'] = timestamps.dt.floor('H')
        posts_per_hour = campaign_posts.groupby('hour').size()
        
        # Burst detection (posts within 10 min windows)
        campaign_posts['minute_10'] = timestamps.dt.floor('10min')
        posts_per_10min = campaign_posts.groupby('minute_10').size()
        max_burst = posts_per_10min.max()
        
        # Inter-post intervals
        sorted_times = timestamps.sort_values()
        intervals = sorted_times.diff().dt.total_seconds().dropna()
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'posts_per_hour': posts_per_hour,
            'posts_per_10min': posts_per_10min,
            'max_burst': max_burst,
            'mean_interval_sec': intervals.mean(),
            'median_interval_sec': intervals.median(),
            'min_interval_sec': intervals.min(),
            'intervals': intervals.values
        }
    
    def analyze_user_network(self, campaign_id):
        """
        Analyze user coordination patterns within a campaign.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            User network analysis
        """
        campaign_posts = self.assignments[self.assignments['campaign_id'] == campaign_id]
        
        if len(campaign_posts) == 0:
            return None
        
        # User posting frequency
        user_post_counts = campaign_posts['user_id'].value_counts()
        
        # Find users posting multiple times (potential bots/coordinators)
        power_users = user_post_counts[user_post_counts > 1]
        
        # Temporal co-posting (users posting within 1 hour of each other)
        campaign_posts['hour'] = pd.to_datetime(campaign_posts['timestamp']).dt.floor('H')
        user_hour_pairs = []
        
        for hour, group in campaign_posts.groupby('hour'):
            users_in_hour = group['user_id'].unique()
            if len(users_in_hour) > 1:
                # Record co-posting
                for i, u1 in enumerate(users_in_hour):
                    for u2 in users_in_hour[i+1:]:
                        user_hour_pairs.append((u1, u2))
        
        # Count co-posting frequency
        from collections import Counter
        coposting_counts = Counter(user_hour_pairs)
        
        return {
            'total_users': len(user_post_counts),
            'power_users': len(power_users),
            'user_post_counts': user_post_counts,
            'power_user_list': power_users.to_dict(),
            'coposting_pairs': coposting_counts.most_common(10)
        }
    
    def visualize_campaign(self, campaign_id, save_path=None):
        """
        Create comprehensive visualization of a campaign.
        
        Args:
            campaign_id: Campaign ID
            save_path: Optional path to save figure
        """
        info = self.get_campaign_info(campaign_id)
        
        if info is None:
            print(f"⚠️  Campaign {campaign_id} not found")
            return
        
        temporal = self.analyze_temporal_pattern(campaign_id)
        user_network = self.analyze_user_network(campaign_id)
        
        # Create figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # Title
        fig.suptitle(f"Campaign {campaign_id} Analysis", 
                    fontsize=16, fontweight='bold')
        
        # 1. Campaign statistics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        stats_text = f"""
Campaign Statistics
{'='*30}
Posts: {info['stats']['n_posts']}
Users: {info['stats']['n_users']}
Posts/User: {info['stats']['posts_per_user']:.2f}
Duration: {info['stats']['time_span_hours']:.1f} hours
Mean Anomaly: {info['stats']['mean_anomaly_score']:.3f}
Coordination: {info['stats']['coordination_score']:.2f}
        """
        ax1.text(0.1, 0.5, stats_text, fontsize=11, 
                verticalalignment='center', family='monospace')
        
        # 2. Temporal pattern (timeline)
        ax2 = fig.add_subplot(gs[0, 1:])
        posts = info['posts']
        timestamps = pd.to_datetime(posts['timestamp'])
        ax2.scatter(timestamps, posts['anomaly_score'], 
                   c=posts['anomaly_score'], cmap='YlOrRd',
                   s=50, alpha=0.6, edgecolors='black')
        ax2.set_title('Post Timeline', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Anomaly Score')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Posts per hour histogram
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(range(len(temporal['posts_per_hour'])), 
               temporal['posts_per_hour'].values,
               color='steelblue', edgecolor='black', alpha=0.7)
        ax3.set_title('Posts per Hour', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Posts')
        
        # 4. Burst detection (10-min windows)
        ax4 = fig.add_subplot(gs[1, 1])
        burst_values = temporal['posts_per_10min'].values
        ax4.hist(burst_values, bins=min(20, len(burst_values)), 
                color='coral', edgecolor='black', alpha=0.7)
        ax4.axvline(temporal['max_burst'], color='red', 
                   linestyle='--', label=f'Max: {temporal["max_burst"]}')
        ax4.set_title('Burst Analysis (10-min windows)', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Posts per 10-min')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Inter-post intervals
        ax5 = fig.add_subplot(gs[1, 2])
        intervals_min = temporal['intervals'] / 60  # Convert to minutes
        ax5.hist(intervals_min[intervals_min < 60], bins=30,  # Show <1 hour
                color='lightgreen', edgecolor='black', alpha=0.7)
        ax5.set_title('Inter-Post Intervals (<1 hour)', 
                     fontsize=12, fontweight='bold')
        ax5.set_xlabel('Minutes Between Posts')
        ax5.set_ylabel('Frequency')
        ax5.axvline(temporal['median_interval_sec']/60, 
                   color='red', linestyle='--', 
                   label=f'Median: {temporal["median_interval_sec"]/60:.1f}min')
        ax5.legend()
        
        # 6. User posting frequency
        ax6 = fig.add_subplot(gs[2, 0])
        user_counts = user_network['user_post_counts']
        ax6.bar(range(min(20, len(user_counts))), 
               user_counts.values[:20],
               color='plum', edgecolor='black', alpha=0.7)
        ax6.set_title('Top 20 Users by Posts', fontsize=12, fontweight='bold')
        ax6.set_xlabel('User Rank')
        ax6.set_ylabel('Posts')
        
        # 7. User post distribution
        ax7 = fig.add_subplot(gs[2, 1])
        post_dist = user_counts.value_counts().sort_index()
        ax7.bar(post_dist.index, post_dist.values, 
               color='skyblue', edgecolor='black', alpha=0.7)
        ax7.set_title('User Post Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Posts per User')
        ax7.set_ylabel('Number of Users')
        
        # 8. Power users highlight
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        power_text = "Top Power Users\n" + "="*25 + "\n"
        for user, count in list(user_network['power_user_list'].items())[:10]:
            power_text += f"{user[:20]:20s}: {count:3d}\n"
        ax8.text(0.1, 0.95, power_text, fontsize=9, 
                verticalalignment='top', family='monospace')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        plt.tight_layout()
        return fig
    
    def compare_campaigns(self, campaign_ids, metric='coordination_score'):
        """
        Compare multiple campaigns on a given metric.
        
        Args:
            campaign_ids: List of campaign IDs
            metric: Metric to compare (default: coordination_score)
        """
        campaigns_data = self.statistics[self.statistics['campaign_id'].isin(campaign_ids)]
        
        if len(campaigns_data) == 0:
            print("⚠️  No campaigns found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Campaign Comparison', fontsize=16, fontweight='bold')
        
        # 1. Primary metric
        axes[0, 0].bar(campaigns_data['campaign_id'].astype(str), 
                      campaigns_data[metric],
                      color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Campaign ID')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Posts vs Users
        axes[0, 1].scatter(campaigns_data['n_users'], 
                          campaigns_data['n_posts'],
                          c=campaigns_data[metric],
                          cmap='YlOrRd', s=200, 
                          edgecolors='black', alpha=0.7)
        axes[0, 1].set_title('Posts vs Users', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Users')
        axes[0, 1].set_ylabel('Posts')
        
        # 3. Time span comparison
        axes[1, 0].bar(campaigns_data['campaign_id'].astype(str), 
                      campaigns_data['time_span_hours'],
                      color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Duration (hours)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Campaign ID')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Anomaly scores
        axes[1, 1].bar(campaigns_data['campaign_id'].astype(str), 
                      campaigns_data['mean_anomaly_score'],
                      color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Mean Anomaly Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Campaign ID')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def export_campaign_report(self, campaign_id, output_path=None):
        """
        Generate a detailed text report for a campaign.
        
        Args:
            campaign_id: Campaign ID
            output_path: Optional file path to save report
        """
        info = self.get_campaign_info(campaign_id)
        
        if info is None:
            print(f"⚠️  Campaign {campaign_id} not found")
            return
        
        temporal = self.analyze_temporal_pattern(campaign_id)
        user_network = self.analyze_user_network(campaign_id)
        
        report = f"""
{'='*80}
CAMPAIGN INVESTIGATION REPORT
Campaign ID: {campaign_id}
{'='*80}

1. OVERVIEW
-----------
Total Posts:           {info['stats']['n_posts']}
Unique Users:          {info['stats']['n_users']}
Posts per User:        {info['stats']['posts_per_user']:.2f}
Duration:              {info['stats']['time_span_hours']:.2f} hours ({info['stats']['time_span_hours']/24:.2f} days)
Mean Anomaly Score:    {info['stats']['mean_anomaly_score']:.4f}
Max Anomaly Score:     {info['stats']['max_anomaly_score']:.4f}
Coordination Score:    {info['stats']['coordination_score']:.4f}

2. TEMPORAL ANALYSIS
--------------------
Start Time:            {temporal['start_time']}
End Time:              {temporal['end_time']}
Duration:              {temporal['duration']}
Maximum Burst:         {temporal['max_burst']} posts in 10 minutes
Mean Interval:         {temporal['mean_interval_sec']:.1f} seconds
Median Interval:       {temporal['median_interval_sec']:.1f} seconds
Minimum Interval:      {temporal['min_interval_sec']:.1f} seconds

⚠️  COORDINATION INDICATORS:
   - {'HIGH' if temporal['max_burst'] > 5 else 'MODERATE' if temporal['max_burst'] > 3 else 'LOW'} burst activity
   - {'RAPID' if temporal['median_interval_sec'] < 300 else 'MODERATE' if temporal['median_interval_sec'] < 1800 else 'SLOW'} posting cadence

3. USER NETWORK ANALYSIS
-------------------------
Total Users:           {user_network['total_users']}
Power Users (>1 post): {user_network['power_users']}
Percentage Power:      {user_network['power_users']/user_network['total_users']*100:.1f}%

Top 10 Most Active Users:
"""
        for user, count in list(user_network['user_post_counts'].items())[:10]:
            report += f"   {user:30s}  {count:3d} posts\n"
        
        report += f"""
Co-Posting Analysis (users posting within same hour):
"""
        if user_network['coposting_pairs']:
            for (u1, u2), count in user_network['coposting_pairs'][:5]:
                report += f"   {u1} ↔ {u2}: {count} times\n"
        else:
            report += "   No significant co-posting detected\n"
        
        report += f"""
⚠️  SUSPICION INDICATORS:
   - {'HIGH' if user_network['power_users']/user_network['total_users'] > 0.5 else 'MODERATE' if user_network['power_users']/user_network['total_users'] > 0.2 else 'LOW'} concentration (few users, many posts)
   - {'SIGNIFICANT' if len(user_network['coposting_pairs']) > 0 else 'NO'} coordinated posting detected

4. RECOMMENDATION
-----------------
"""
        
        # Determine threat level
        threat_score = 0
        if temporal['max_burst'] > 5:
            threat_score += 3
        elif temporal['max_burst'] > 3:
            threat_score += 1
        
        if user_network['power_users'] / user_network['total_users'] > 0.5:
            threat_score += 3
        elif user_network['power_users'] / user_network['total_users'] > 0.2:
            threat_score += 1
        
        if info['stats']['mean_anomaly_score'] > 0.7:
            threat_score += 3
        elif info['stats']['mean_anomaly_score'] > 0.5:
            threat_score += 1
        
        if threat_score >= 7:
            threat_level = "CRITICAL"
            recommendation = "Immediate investigation required. High likelihood of coordinated campaign."
        elif threat_score >= 4:
            threat_level = "HIGH"
            recommendation = "Priority investigation recommended. Multiple red flags detected."
        elif threat_score >= 2:
            threat_level = "MEDIUM"
            recommendation = "Monitor closely. Some suspicious patterns detected."
        else:
            threat_level = "LOW"
            recommendation = "Standard monitoring. Limited coordination indicators."
        
        report += f"Threat Level: {threat_level}\n"
        report += f"Action: {recommendation}\n"
        report += f"\n{'='*80}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to {output_path}")
        
        return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CAMPAIGN INVESTIGATOR - ANALYSIS TOOL")
    print("="*80)
    
    # Initialize investigator
    investigator = CampaignInvestigator("campaign_detection_results")
    
    # Get top campaigns
    top_campaigns = investigator.statistics.nlargest(5, 'coordination_score')
    
    print(f"\n🎯 Top 5 Coordinated Campaigns:")
    print(top_campaigns[['campaign_id', 'n_posts', 'n_users', 'coordination_score']])
    
    # Investigate top campaign
    if len(top_campaigns) > 0:
        top_id = top_campaigns.iloc[0]['campaign_id']
        
        print(f"\n🔍 Investigating Campaign {top_id}...")
        
        # Generate report
        report = investigator.export_campaign_report(
            top_id, 
            f"campaign_detection_results/campaign_{top_id}_report.txt"
        )
        
        print("\n" + report)
        
        # Create visualization
        investigator.visualize_campaign(
            top_id, 
            f"campaign_detection_results/campaign_{top_id}_analysis.png"
        )
        
    print("\n" + "="*80)
    print("✅ INVESTIGATION COMPLETE!")
    print("="*80)