"""
INTERACTIVE EXPLAINABILITY DASHBOARD
=====================================
Generates an interactive HTML dashboard for analysts to explore
suspicious posts with detailed explanations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("GENERATING INTERACTIVE EXPLAINABILITY DASHBOARD")
print("="*80)

# Load explainability results
output_dir = Path("explainability_results")
explanations = pd.read_csv(output_dir / "post_explanations.csv")
suspicious_phrases = pd.read_csv(output_dir / "suspicious_phrases.csv")
user_risks = pd.read_csv(output_dir / "user_risk_scores.csv")

# Load original data for post content
df = pd.read_pickle("Dataset/twitter/df_with_all_features.pkl")
df['post_id'] = df['post_id'].astype(int)

# Merge
explanations_full = explanations.merge(df, on='post_id', how='left')

print(f"✅ Loaded {len(explanations)} post explanations")

# ============================================================================
# GENERATE HTML DASHBOARD
# ============================================================================

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Suspicious Content Explainability Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .search-box {{
            width: 100%;
            padding: 15px;
            font-size: 1em;
            border: 2px solid #667eea;
            border-radius: 10px;
            margin-bottom: 30px;
            outline: none;
            transition: border-color 0.3s;
        }}
        
        .search-box:focus {{
            border-color: #764ba2;
        }}
        
        .post-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s;
        }}
        
        .post-card:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
        }}
        
        .post-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .post-id {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}
        
        .suspicion-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        
        .badge-very-high {{ background: #dc3545; }}
        .badge-high {{ background: #fd7e14; }}
        .badge-medium {{ background: #ffc107; color: #333; }}
        .badge-low {{ background: #28a745; }}
        
        .post-text {{
            font-size: 1.1em;
            line-height: 1.6;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .highlighted-word {{
            background: #ff6b6b;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        .detection-methods {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .method-tag {{
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85em;
        }}
        
        .explanation {{
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }}
        
        .user-risk {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 12px;
            background: #fff3cd;
            border-radius: 6px;
            font-size: 0.9em;
        }}
        
        .filter-controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 10px 20px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        
        .filter-btn:hover, .filter-btn.active {{
            background: #667eea;
            color: white;
        }}
        
        .top-phrases {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }}
        
        .phrase-tag {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Suspicious Content Explainability Dashboard</h1>
            <p>AI-Powered Detection with Human-Readable Explanations</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Analyzed</h3>
                <div class="value">{total_posts}</div>
            </div>
            <div class="stat-card">
                <h3>High Confidence</h3>
                <div class="value">{high_conf}</div>
            </div>
            <div class="stat-card">
                <h3>Detection Methods</h3>
                <div class="value">4</div>
            </div>
            <div class="stat-card">
                <h3>Avg Suspicion Score</h3>
                <div class="value">{avg_suspicion:.2f}</div>
            </div>
        </div>
        
        <div class="content">
            <h2 style="margin-bottom: 20px;">🎯 Top Suspicious Phrases Detected</h2>
            <div class="top-phrases">
                {phrase_tags}
            </div>
            
            <h2 style="margin-top: 40px; margin-bottom: 20px;">📋 Suspicious Post Explanations</h2>
            
            <input type="text" class="search-box" id="searchBox" 
                   placeholder="Search by post ID, text, or username..." 
                   onkeyup="filterPosts()">
            
            <div class="filter-controls">
                <button class="filter-btn active" onclick="filterByScore('all')">All Posts</button>
                <button class="filter-btn" onclick="filterByScore('very-high')">Very High Risk</button>
                <button class="filter-btn" onclick="filterByScore('high')">High Risk</button>
                <button class="filter-btn" onclick="filterByScore('medium')">Medium Risk</button>
            </div>
            
            <div id="postsContainer">
                {post_cards}
            </div>
        </div>
    </div>
    
    <script>
        function filterPosts() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const posts = document.getElementsByClassName('post-card');
            
            for (let post of posts) {{
                const text = post.textContent.toLowerCase();
                post.style.display = text.includes(searchTerm) ? 'block' : 'none';
            }}
        }}
        
        function filterByScore(level) {{
            const posts = document.getElementsByClassName('post-card');
            const buttons = document.getElementsByClassName('filter-btn');
            
            // Update active button
            for (let btn of buttons) {{
                btn.classList.remove('active');
            }}
            event.target.classList.add('active');
            
            // Filter posts
            for (let post of posts) {{
                if (level === 'all') {{
                    post.style.display = 'block';
                }} else {{
                    const badge = post.querySelector('.suspicion-badge');
                    const badgeClass = badge.className;
                    
                    if (badgeClass.includes(level)) {{
                        post.style.display = 'block';
                    }} else {{
                        post.style.display = 'none';
                    }}
                }}
            }}
        }}
    </script>
</body>
</html>
"""

# The rest of your script (phrase_tags_html, post_cards_html, filling template, saving) remains unchanged.
