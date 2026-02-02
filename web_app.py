"""
FLASK WEB INTERFACE FOR FAKE NEWS DETECTION
============================================
Simple web UI for users to submit posts and get explainability results.

Run with: python web_app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
from datetime import datetime
import json

# Import the detector
from interactive_detection_sys import InteractiveFakeNewsDetector

app = Flask(__name__)

# Initialize detector (load once at startup)
print("Initializing detection system...")
detector = InteractiveFakeNewsDetector()
print("✅ System ready!")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        
        .content { padding: 40px; }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .tab {
            padding: 15px 30px;
            background: white;
            border: none;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            color: #667eea;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab:hover { background: #f8f9fa; }
        .tab.active { border-bottom-color: #667eea; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
        }
        
        .form-group textarea,
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1em;
            font-family: inherit;
        }
        
        .form-group textarea {
            min-height: 150px;
            resize: vertical;
        }
        
        .form-group input[type="file"] {
            padding: 10px;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s;
        }
        
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        
        .results.show { display: block; }
        
        .prediction-badge {
            display: inline-block;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.5em;
            font-weight: bold;
            color: white;
            margin: 20px 0;
        }
        
        .prediction-fake {
            background: linear-gradient(135deg, #dc3545, #c82333);
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
        }
        
        .prediction-real {
            background: linear-gradient(135deg, #28a745, #218838);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        }
        
        .risk-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            margin: 10px 0;
        }
        
        .risk-critical { background: #dc3545; }
        .risk-high { background: #fd7e14; }
        .risk-moderate { background: #ffc107; color: #333; }
        .risk-low { background: #28a745; }
        
        .fusion-chart {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        
        .fusion-bar {
            flex: 1;
            text-align: center;
        }
        
        .fusion-bar-inner {
            background: #667eea;
            height: 30px;
            border-radius: 5px;
            transition: width 0.5s;
        }
        
        .suspicious-phrase {
            display: inline-block;
            background: #ff6b6b;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px;
            font-weight: bold;
        }
        
        .campaign-alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        
        .loading.show { display: block; }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Fake News Detection System</h1>
            <p>AI-Powered Multimodal Detection with Explainability</p>
        </div>
        
        <div class="content">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('single')">
                    Single Post Detection
                </button>
                <button class="tab" onclick="switchTab('multiple')">
                    Multiple Posts (Campaign Detection)
                </button>
            </div>
            
            <!-- Single Post Tab -->
            <div id="single-tab" class="tab-content active">
                <form id="single-form" onsubmit="analyzeSinglePost(event)">
                    <div class="form-group">
                        <label>Post Text *</label>
                        <textarea name="text" placeholder="Enter the post text here..." required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>Image (optional)</label>
                        <input type="file" name="image" accept="image/*">
                    </div>
                    
                    <div class="form-group">
                        <label>User Metadata (optional)</label>
                        <div class="metadata-grid">
                            <input type="text" name="username" placeholder="Username">
                            <input type="number" name="followers" placeholder="Followers">
                            <input type="number" name="following" placeholder="Following">
                            <input type="number" name="posts_count" placeholder="Posts Count">
                            <input type="number" name="account_age" placeholder="Account Age (days)">
                        </div>
                    </div>
                    
                    <button type="submit" class="btn">Analyze Post</button>
                </form>
                
                <div class="loading" id="single-loading">
                    <div class="spinner"></div>
                    <p>Analyzing post...</p>
                </div>
                
                <div class="results" id="single-results"></div>
            </div>
            
            <!-- Multiple Posts Tab -->
            <div id="multiple-tab" class="tab-content">
                <form id="multiple-form" onsubmit="analyzeMultiplePosts(event)">
                    <div class="form-group">
                        <label>Number of Posts</label>
                        <input type="number" id="num-posts" min="2" max="10" value="3" 
                               onchange="generatePostInputs()">
                    </div>
                    
                    <div id="posts-container"></div>
                    
                    <button type="submit" class="btn">Analyze Posts & Detect Campaigns</button>
                </form>
                
                <div class="loading" id="multiple-loading">
                    <div class="spinner"></div>
                    <p>Analyzing posts and detecting campaigns...</p>
                </div>
                
                <div class="results" id="multiple-results"></div>
            </div>
        </div>
    </div>
    
    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
        }
        
        function generatePostInputs() {
            const num = document.getElementById('num-posts').value;
            const container = document.getElementById('posts-container');
            
            container.innerHTML = '';
            for (let i = 1; i <= num; i++) {
                container.innerHTML += `
                    <div class="form-group" style="border: 1px solid #e9ecef; padding: 15px; margin: 10px 0; border-radius: 8px;">
                        <h4>Post ${i}</h4>
                        <textarea name="text_${i}" placeholder="Post text..." required style="margin-top: 10px;"></textarea>
                        <input type="text" name="username_${i}" placeholder="Username" style="margin-top: 10px;">
                    </div>
                `;
            }
        }
        
        // Initialize
        generatePostInputs();
        
        async function analyzeSinglePost(event) {
            event.preventDefault();
            
            const form = event.target;
            const formData = new FormData(form);
            
            // Show loading
            document.getElementById('single-loading').classList.add('show');
            document.getElementById('single-results').classList.remove('show');
            
            // Prepare data
            const data = {
                text: formData.get('text'),
                metadata: {
                    username: formData.get('username') || 'unknown',
                    followers: parseInt(formData.get('followers')) || 0,
                    following: parseInt(formData.get('following')) || 0,
                    posts_count: parseInt(formData.get('posts_count')) || 0,
                    account_age_days: parseInt(formData.get('account_age')) || 0
                }
            };
            
            try {
                const response = await fetch('/api/detect_single', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                displaySingleResult(result);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('single-loading').classList.remove('show');
            }
        }
        
        function displaySingleResult(result) {
            const container = document.getElementById('single-results');
            
            const riskClass = 'risk-' + result.risk_level.toLowerCase();
            
            let html = `
                <h2>Detection Results</h2>
                <div class="risk-badge ${riskClass}">${result.risk_level} RISK</div>
                
                <h3 style="margin-top: 20px;">Suspicion Score: ${(result.suspicion_score * 100).toFixed(1)}%</h3>
                <p>Confidence Level: <strong>${result.confidence_level}</strong></p>
                
                <h3 style="margin-top: 30px;">Modality Contributions</h3>
                <div class="fusion-chart">
                    <div class="fusion-bar">
                        <div class="fusion-bar-inner" style="width: ${result.fusion_weights.text * 100}%"></div>
                        <p>Text: ${(result.fusion_weights.text * 100).toFixed(1)}%</p>
                    </div>
                    <div class="fusion-bar">
                        <div class="fusion-bar-inner" style="width: ${result.fusion_weights.image * 100}%"></div>
                        <p>Image: ${(result.fusion_weights.image * 100).toFixed(1)}%</p>
                    </div>
                    <div class="fusion-bar">
                        <div class="fusion-bar-inner" style="width: ${result.fusion_weights.metadata * 100}%"></div>
                        <p>Metadata: ${(result.fusion_weights.metadata * 100).toFixed(1)}%</p>
                    </div>
                </div>
                
                <p><strong>Dominant Modality:</strong> ${result.dominant_modality}</p>
            `;
            
            if (result.suspicious_phrases && result.suspicious_phrases.length > 0) {
                html += `
                    <h3 style="margin-top: 30px;">Suspicious Phrases Detected</h3>
                    <div>
                `;
                result.suspicious_phrases.forEach(([phrase, score]) => {
                    html += `<span class="suspicious-phrase">${phrase}</span>`;
                });
                html += `</div>`;
            }
            
            html += `
                <h3 style="margin-top: 30px;">Explanation</h3>
                <p style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea;">
                    ${result.summary}
                </p>
            `;
            
            container.innerHTML = html;
            container.classList.add('show');
        }
        
        async function analyzeMultiplePosts(event) {
            event.preventDefault();
            
            const form = event.target;
            const formData = new FormData(form);
            const num = document.getElementById('num-posts').value;
            
            // Show loading
            document.getElementById('multiple-loading').classList.add('show');
            document.getElementById('multiple-results').classList.remove('show');
            
            // Prepare data
            const posts = [];
            for (let i = 1; i <= num; i++) {
                posts.push({
                    text: formData.get(`text_${i}`),
                    metadata: {
                        username: formData.get(`username_${i}`) || `user_${i}`
                    }
                });
            }
            
            try {
                const response = await fetch('/api/detect_multiple', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({posts})
                });
                
                const result = await response.json();
                displayMultipleResults(result);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('multiple-loading').classList.remove('show');
            }
        }
        
        function displayMultipleResults(result) {
            const container = document.getElementById('multiple-results');
            
            let html = `
                <h2>Batch Analysis Results</h2>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                    <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <h3>${result.summary.total_posts}</h3>
                        <p>Total Posts</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #dc3545;">${result.summary.suspicious_posts}</h3>
                        <p>Fake Posts</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #28a745;">${result.summary.total_posts - result.summary.suspicious_posts}</h3>
                        <p>Real Posts</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <h3>${(result.summary.avg_suspicion_score * 100).toFixed(1)}%</h3>
                        <p>Avg Suspicion</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="color: #ffc107;">${result.summary.num_campaigns}</h3>
                        <p>Campaigns</p>
                    </div>
                </div>
            `;
            
            if (result.campaigns && result.campaigns.num_campaigns > 0) {
                html += `
                    <div class="campaign-alert">
                        <h3>🚨 Coordination Campaign Detected!</h3>
                        <p>${result.campaigns.summary}</p>
                `;
                
                result.campaigns.campaigns.forEach((camp, i) => {
                    html += `
                        <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 5px;">
                            <strong>Campaign ${i + 1}:</strong>
                            ${camp.num_posts} posts by ${camp.num_users} users
                            (Avg suspicion: ${(camp.avg_suspicion * 100).toFixed(1)}%)
                            ${camp.is_coordinated ? '<span style="color: #dc3545;">⚠️ COORDINATED</span>' : ''}
                        </div>
                    `;
                });
                
                html += `</div>`;
            }
            
            html += `<h3 style="margin-top: 30px;">Individual Post Results</h3>`;
            
            result.posts.forEach((post, i) => {
                const riskClass = 'risk-' + post.risk_level.toLowerCase();
                const predictionClass = post.is_fake ? 'prediction-fake' : 'prediction-real';
                const predictionIcon = post.is_fake ? '❌' : '✅';
                
                html += `
                    <div style="background: white; padding: 20px; margin: 10px 0; border-radius: 10px;">
                        <h4>Post ${i + 1}</h4>
                        <div class="prediction-badge ${predictionClass}" style="font-size: 1em; padding: 8px 16px;">
                            ${predictionIcon} ${post.prediction}
                        </div>
                        <span class="risk-badge ${riskClass}">${post.risk_level}</span>
                        <p>Suspicion: ${(post.suspicion_score * 100).toFixed(1)}%</p>
                        <p style="margin-top: 10px; font-size: 0.9em;">${post.summary}</p>
                    </div>
                `;
            });
            
            container.innerHTML = html;
            container.classList.add('show');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect_single', methods=['POST'])
def detect_single():
    data = request.json
    
    result = detector.detect_single_post(
        text=data['text'],
        image=None,
        metadata=data.get('metadata'),
        timestamp=datetime.now()
    )
    
    return jsonify(result)

@app.route('/api/detect_multiple', methods=['POST'])
def detect_multiple():
    data = request.json
    posts = data['posts']
    
    # Add timestamps
    for post in posts:
        post['timestamp'] = datetime.now()
    
    result = detector.detect_multiple_posts(posts, detect_campaigns=True)
    
    return jsonify(result)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING WEB SERVER")
    print("="*80)
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5000)