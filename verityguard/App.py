"""
FAKE NEWS DETECTOR - FLASK API
===============================
RESTful API for real-time fake news detection with support for:
- Single post analysis (text + optional image)
- Batch CSV processing
- Result storage and retrieval
- Health monitoring
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
import io
import base64
from PIL import Image
import threading

init_lock = threading.Lock()

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------
# APP SETUP
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# GLOBALS (LAZY INIT)
# ------------------------------------------------------------
detector = None
post_pattern_pipeline = None
models_ready = False

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# ------------------------------------------------------------
# LAZY MODEL INITIALIZATION (RAILWAY SAFE)
# ------------------------------------------------------------
def load_models():
    global detector, post_pattern_pipeline, models_ready
    if models_ready:
        return

    with init_lock:
        if models_ready:
            return

        # Apply memory optimizations BEFORE loading models
        try:
            from memory_utils import optimize_torch_memory
            optimize_torch_memory()
        except Exception as e:
            print(f"⚠️ Could not apply memory optimizations: {e}")

        print("🤗 Downloading models from Hugging Face...")
        from download_models_hf import download_from_huggingface
        download_from_huggingface()

        print("🚀 Initializing Fake News Detector...")
        
        # Smart import: try relative first (for production), fall back to absolute (for local dev)
        try:
            from .standalone_detector import StandaloneFakeNewsDetector
        except ImportError:
            from standalone_detector import StandaloneFakeNewsDetector
        
        from inference_post_patterns import ComprehensiveInferencePipeline

        detector = StandaloneFakeNewsDetector()
        post_pattern_pipeline = ComprehensiveInferencePipeline()

        models_ready = True
        print("✅ Models and detectors ready")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def load_results(limit=100):
    results = []
    for fp in sorted(Path(app.config['RESULTS_FOLDER']).glob("*.json"),
                     key=os.path.getmtime, reverse=True)[:limit]:
        with open(fp, 'r') as f:
            results.append(json.load(f))
    return results


def require_models():
    if not models_ready:
        return jsonify({'success': False, 'error': 'Models are still loading'}), 503
    return None

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    # Don't load models on health check
    return jsonify({
        'status': 'healthy',
        'models_ready': models_ready
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict_api():
    load_models()  # Ensure models are loaded

    if not models_ready:
        return require_models()

    try:
        text = ""
        username = "anonymous"
        image_path = None

        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip()
            username = data.get("username", "anonymous")
            image_base64 = data.get("image_base64")

            if image_base64:
                image_data = base64.b64decode(image_base64.split(",")[-1])
                image = Image.open(io.BytesIO(image_data))
                filename = f"{uuid.uuid4()}.png"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
        else:
            text = request.form.get("text", "").strip()
            username = request.form.get("username", "anonymous")
            image = request.files.get("image")

            if image:
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)

        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400

        # Capture detector output
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        detector.predict(text=text, image_path=image_path, username=username)
        detector_output = sys.stdout.getvalue()
        sys.stdout = io.StringIO()
        post_pattern_pipeline.process_post(text=text, image_path=image_path, username=username)
        pattern_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            'success': True,
            'detector_output': detector_output,
            'pattern_output': pattern_output
        }), 200

    except Exception as e:
        sys.stdout = old_stdout
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    return jsonify({
        'success': True,
        'results': load_results(int(request.args.get('limit', 100)))
    }), 200


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    results = load_results(1000)
    return jsonify({
        'success': True,
        'total': len(results),
        'fake': sum(1 for r in results if r.get('verdict') == 'FAKE'),
        'real': sum(1 for r in results if r.get('verdict') == 'REAL')
    }), 200


# ------------------------------------------------------------
# ERROR HANDLERS
# ------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)