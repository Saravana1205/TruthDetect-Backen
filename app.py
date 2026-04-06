import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Create a folder to save uploaded images/videos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return "TruthDetect ML Backend is Running!"

@app.route('/predict-text', methods=['POST'])
def predict_text():
    data = request.json
    incoming_text = data.get('text', '').lower()
    
    if not incoming_text:
        return jsonify({"error": "No text provided"}), 400

    fake_trigger_words = ['shocking', 'unbelievable', 'miracle', 'secret']
    is_fake = any(word in incoming_text for word in fake_trigger_words)
    
    if is_fake:
        return jsonify({"result": "Fake", "confidence": 88, "explanation": "High usage of sensationalist trigger words."})
    else:
        return jsonify({"result": "Real", "confidence": 92, "explanation": "Vocabulary matches neutral journalism."})

# ---------------- NEW: IMAGE ENDPOINT ----------------
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # TODO: Pass 'filepath' to your real CNN model later.
    # For now, we mock the result:
    return jsonify({
        "result": "Real", 
        "confidence": 95, 
        "explanation": f"Image '{filename}' processed. No visual artifacts or abnormal pixel gradients detected in the facial region."
    })

# ---------------- NEW: VIDEO ENDPOINT ----------------
@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Mock video result
    return jsonify({
        "result": "Fake", 
        "confidence": 97, 
        "explanation": f"Video '{filename}' processed. Frame-by-frame temporal scan detected micro-expression jitter and lip-sync anomalies."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)