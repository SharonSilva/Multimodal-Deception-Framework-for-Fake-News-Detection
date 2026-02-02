"""
FAKE NEWS DETECTOR - FLASK API
===============================
RESTful API for real-time fake news detection with support for:
- Single post analysis (text + optional image)
- Batch CSV processing
- Result storage and retrieval
- Health monitoring
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import csv
import uuid
from datetime import datetime
from pathlib import Path
import io
import base64
from PIL import Image
from  flask import Response
from download_models_hf import download_from_huggingface
download_from_huggingface()

# Import the standalone detector
from standalone_detector import StandaloneFakeNewsDetector
from inference_post_patterns import ComprehensiveInferencePipeline

detector = StandaloneFakeNewsDetector()
post_pattern_pipeline = ComprehensiveInferencePipeline()


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['ALLOWED_CSV_EXTENSIONS'] = {'csv'}

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# # Initialize detector (singleton)
# print("🚀 Initializing Fake News Detector...")
# detector = StandaloneFakeNewsDetector()

def get_detector():
    """Lazy load detector on first request"""
    global detector
    if detector is None:
        detector = StandaloneFakeNewsDetector()
    return detector


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def save_result(result):
    """Save prediction result to JSON file"""
    result_id = result['post_id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result_id}_{timestamp}.json"
    filepath = Path(app.config['RESULTS_FOLDER']) / filename
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    return str(filepath)


def load_results(limit=100):
    """Load recent results from storage"""
    results_dir = Path(app.config['RESULTS_FOLDER'])
    result_files = sorted(results_dir.glob('*.json'), 
                         key=os.path.getmtime, 
                         reverse=True)[:limit]
    
    results = []
    for filepath in result_files:
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    
    return results


# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(detector.device),
        'emotion_model_loaded': detector.emotion_model is not None,
        'fusion_model_loaded': detector.fusion_model is not None
    }), 200

@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        # -----------------------------
        # 1️⃣ Get text & username
        # -----------------------------
        text = ""
        username = "anonymous"
        image_path = None

        # Check if JSON payload
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip()
            username = data.get("username", "anonymous")
            image_base64 = data.get("image_base64", None)
            if image_base64:
                import base64
                from PIL import Image
                import uuid
                try:
                    image_data = base64.b64decode(image_base64.split(",")[-1])
                    image = Image.open(io.BytesIO(image_data))
                    filename = f"{uuid.uuid4()}.png"
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    image.save(image_path)
                except Exception as e:
                    return jsonify({'success': False, 'error': f'Invalid image data: {str(e)}'}), 400
        else:
            # Form-data (from HTML form)
            text = request.form.get("text", "").strip()
            username = request.form.get("username", "anonymous")
            image = request.files.get("image")
            if image:
                from werkzeug.utils import secure_filename
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image.save(image_path)

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        # -----------------------------
        # 2️⃣ Run Fake News Detector
        # -----------------------------
        import traceback
        detector_output = ""
        pattern_output = ""

        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            detector.predict(text=text, image_path=image_path, username=username)
            detector_output = sys.stdout.getvalue()
        except Exception as e:
            traceback.print_exc()
            detector_output = f"ERROR in detector: {str(e)}"
        finally:
            sys.stdout = old_stdout

        # -----------------------------
        # 3️⃣ Run Post Pattern Pipeline
        # -----------------------------
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            post_pattern_pipeline.process_post(text=text, image_path=image_path, username=username)
            pattern_output = sys.stdout.getvalue()
        except Exception as e:
            traceback.print_exc()
            pattern_output = f"ERROR in pattern pipeline: {str(e)}"
        finally:
            sys.stdout = old_stdout

        # -----------------------------
        # 4️⃣ Cleanup temp image
        # -----------------------------
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        # -----------------------------
        # 5️⃣ Return JSON with both outputs
        # -----------------------------
        return jsonify({
            'success': True,
            'detector_output': detector_output,
            'pattern_output': pattern_output
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    """
    Analyze a post with base64-encoded image
    
    Request (JSON):
        - text: str (required)
        - image_base64: str (optional) - Base64-encoded image
        - username: str (optional)
    
    Response: Same as /api/predict
    """
    try:
        data = request.get_json()
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text content is required'}), 400
        
        username = data.get('username', 'anonymous')
        
        # Handle base64 image
        image_path = None
        if 'image_base64' in data and data['image_base64']:
            try:
                # Decode base64 image
                image_data = base64.b64decode(data['image_base64'].split(',')[-1])
                image = Image.open(io.BytesIO(image_data))
                
                # Save temporarily
                filename = f"{uuid.uuid4()}.png"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Run prediction
        det = detector
        result = det.predict(text=text, image_path=image_path, username=username)
        
        # Save result
        result['timestamp'] = datetime.now().isoformat()
        save_result(result)
        
        # Clean up
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        return jsonify({
            'success': True,
            'verdict': result['verdict'],
            'score': result['score'],
            'confidence': result['confidence'],
            'post_id': result['post_id'],
            'detailed_analysis': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_process():
    """
    Process a batch of posts from CSV file
    
    Request (multipart/form-data):
        - file: CSV file with columns: text, image_path (optional), username (optional)
    
    Response:
        - results: list of predictions
        - summary: statistics
        - download_url: URL to download full results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        csv_file = request.files['file']
        if not csv_file.filename or not allowed_file(csv_file.filename, 
                                                     app.config['ALLOWED_CSV_EXTENSIONS']):
            return jsonify({'error': 'Invalid file type. Must be CSV'}), 400
        
        # Read CSV
        csv_content = csv_file.read().decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        # Process each row
        det = detector
        results = []
        stats = {'total': 0, 'fake': 0, 'real': 0, 'unknown': 0}
        
        for row in csv_reader:
            text = row.get('text', '').strip()
            if not text:
                continue
            
            username = row.get('username', 'anonymous')
            image_path = row.get('image_path', None)
            
            # Validate image path
            if image_path and not os.path.exists(image_path):
                image_path = None
            
            # Run prediction
            result = det.predict(text=text, image_path=image_path, username=username)
            result['timestamp'] = datetime.now().isoformat()
            
            results.append(result)
            stats['total'] += 1
            stats[result['verdict'].lower()] = stats.get(result['verdict'].lower(), 0) + 1
        
        # Save batch results
        batch_id = uuid.uuid4().hex[:8]
        batch_filename = f"batch_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        batch_filepath = Path(app.config['RESULTS_FOLDER']) / batch_filename
        
        with open(batch_filepath, 'w') as f:
            json.dump({
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'statistics': stats
            }, f, indent=2)
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'results': results,
            'statistics': stats,
            'download_url': f'/api/batch/{batch_id}/download'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch/<batch_id>/download', methods=['GET'])
def download_batch(batch_id):
    """Download batch results as JSON"""
    try:
        # Find batch file
        results_dir = Path(app.config['RESULTS_FOLDER'])
        batch_files = list(results_dir.glob(f'batch_{batch_id}_*.json'))
        
        if not batch_files:
            return jsonify({'error': 'Batch not found'}), 404
        
        return send_file(batch_files[0], 
                        as_attachment=True,
                        download_name=f'batch_{batch_id}_results.json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """
    Get recent analysis results
    
    Query params:
        - limit: int (default 100) - Number of results to return
    
    Response:
        - results: list of recent predictions
        - count: total results returned
    """
    try:
        limit = int(request.args.get('limit', 100))
        results = load_results(limit=limit)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """
    Get overall detection statistics
    
    Response:
        - total_analyzed: int
        - fake_count: int
        - real_count: int
        - average_confidence: float
        - recent_trends: dict
    """
    try:
        results = load_results(limit=1000)
        
        stats = {
            'total_analyzed': len(results),
            'fake_count': sum(1 for r in results if r['verdict'] == 'FAKE'),
            'real_count': sum(1 for r in results if r['verdict'] == 'REAL'),
            'unknown_count': sum(1 for r in results if r['verdict'] == 'UNKNOWN'),
            'average_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
            'average_score': sum(r['score'] for r in results) / len(results) if results else 0
        }
        
        # Calculate recent trends (last 24 hours vs previous)
        now = datetime.now()
        recent_results = [r for r in results 
                         if 'timestamp' in r and 
                         (now - datetime.fromisoformat(r['timestamp'])).days < 1]
        
        stats['recent_trends'] = {
            'last_24h_count': len(recent_results),
            'last_24h_fake': sum(1 for r in recent_results if r['verdict'] == 'FAKE'),
            'last_24h_real': sum(1 for r in recent_results if r['verdict'] == 'REAL')
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FAKE NEWS DETECTOR API SERVER")
    print("="*70)
    print("\n📍 Server will start at: http://localhost:5001")
    print("\n📚 API Endpoints:")
    print("   • GET  /                      - Web interface")
    print("   • GET  /api/health            - Health check")
    print("   • POST /api/predict           - Single post analysis")
    print("   • POST /api/predict/base64    - Analysis with base64 image")
    print("   • POST /api/batch             - Batch CSV processing")
    print("   • GET  /api/batch/<id>/download - Download batch results")
    print("   • GET  /api/results           - Get recent results")
    print("   • GET  /api/stats             - Get statistics")
    print("\n" + "="*70 + "\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(
        debug=False,
        use_reloader=False,   # 🔥 REQUIRED
        host="0.0.0.0",
        port=port
    )