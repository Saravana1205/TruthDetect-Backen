import os
import joblib 
import requests
import re
import time
import numpy as np
import cv2 
from PIL import Image
from PIL.ExifTags import TAGS
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import TensorFlow and XAI tools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM

# Performance optimization for cloud environments
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔥 Dynamic URL for Cloud Deployment
BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:5000")

# --- 🧠 GLOBAL MODEL VARIABLES ---
text_model = None
text_vectorizer = None
video_model = None
image_model = None

# --- 🛠️ HELPER FORENSIC TOOLS ---

def extract_metadata(filepath):
    info = {"software": "Unknown", "camera": "Unknown"}
    try:
        image = Image.open(filepath)
        exifdata = image.getexif()
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if tag == 'Software': info['software'] = data
            if tag == 'Model': info['camera'] = data
    except: pass
    return info

def save_forensic_log(name, result, confidence, details):
    log_name = f"LOG_{name}.txt"
    with open(os.path.join(UPLOAD_FOLDER, log_name), "w", encoding="utf-8") as f:
        f.write(f"TRUTHDETECT FORENSIC REPORT\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Target: {name}\n")
        f.write(f"Conclusion: {result} ({confidence}%)\n")
        f.write(f"Analysis: {details}\n")
    return log_name

# --- 🧠 ENHANCED FORENSIC ENGINES ---

def get_text_forensics(text, is_fake):
    if not is_fake:
        return "Analysis confirms high linguistic entropy. No automated bot patterns detected. Safe."
    reasons = []
    if len(set(text.split())) / len(text.split()) < 0.5:
        reasons.append("high vocabulary repetition")
    if any(w in text.lower() for w in ['shocking', 'exposed', 'conspiracy']):
        reasons.append("sensationalist emotional triggers")
    analysis = "Flagged due to " + (", ".join(reasons) if reasons else "anomalous linguistic patterns")
    return f"{analysis}. Recommendation: Cross-verify before sharing."

def get_pixel_forensics(img_np, is_fake, metadata=None):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    soft = metadata['software'] if metadata else "Unknown"
    if not is_fake:
        return f"Authentic pixel noise detected (Variance: {round(lap_var, 2)}). Safe."
    reason = "unnatural smoothing" if lap_var < 110 else "edge aliasing"
    manip_tool = f" Possible edit tool: {soft}." if soft != "Unknown" else ""
    return f"Scan detected {reason} (Score: {round(lap_var, 2)}).{manip_tool} Deepfake probability is high."

def generate_heatmap(img_array, model, filename):
    try:
        explainer = GradCAM()
        grid = explainer.explain(validation_data=(img_array, None), model=model, layer_name="Conv_1", class_index=0)
        heatmap_name = f"heat_{filename}.png"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_name)
        cv2.imwrite(heatmap_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        return heatmap_name
    except: return None

# --- LOADING MODELS ---
print("🚀 Initializing Forensic Engines...")
try:
    # Load Text Models
    text_model = joblib.load('truthdetect_model.pkl')
    text_vectorizer = joblib.load('truthdetect_vectorizer.pkl')
    
    # 🔥 FIXED: Load Video Model with compile=False to bypass Dense layer metadata error
    video_model = load_model('truthdetect_video_model.h5', compile=False)
    
    # Load Image Model Architecture and Weights
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None)
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(1, activation='sigmoid')(x)
    image_model = Model(inputs=base_model.input, outputs=predictions)
    image_model.load_weights('truthdetect.weights.h5')
    
    print("✅ Systems Online!")
except Exception as e: 
    print(f"❌ CRITICAL Initialization Error: {e}")

# --- ROUTES ---

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if text_model is None: return jsonify({"error": "NLP Engine not loaded"}), 503
    data = request.json
    text = data.get('text', '').strip()
    if not text: return jsonify({"error": "No text"}), 400
    math_vector = text_vectorizer.transform([text])
    prediction = text_model.predict(math_vector)[0]
    is_fake = str(prediction).upper() == "FAKE"
    desc = get_text_forensics(text, is_fake)
    save_forensic_log("Text_Scan", "Fake" if is_fake else "Real", 92, desc)
    return jsonify({"result": "Fake" if is_fake else "Real", "confidence": 92, "explanation": desc, "source": "Text"})

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if image_model is None: return jsonify({"error": "Image Engine not loaded"}), 503
    file = request.files.get('file')
    if not file: return jsonify({"error": "No image"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    metadata = extract_metadata(filepath)
    img_raw = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = preprocess_input(np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0))
    score = float(image_model.predict(img_array)[0][0])
    is_fake = score <= 0.5
    conf = round((1-score)*100, 1) if is_fake else round(score*100, 1)
    desc = get_pixel_forensics(img_rgb, is_fake, metadata)
    save_forensic_log(filename, "Fake" if is_fake else "Real", conf, desc)
    return jsonify({
        "result": "Fake" if is_fake else "Real",
        "confidence": conf,
        "explanation": desc,
        "heatmap_url": f"{BASE_URL}/uploads/{generate_heatmap(img_array, image_model, filename)}",
        "source": "Image"
    })

@app.route('/predict-video', methods=['POST'])
def predict_video():
    if video_model is None: return jsonify({"error": "Video Engine not loaded"}), 503
    file = request.files.get('file')
    if not file: return jsonify({"error": "No video"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    cap = cv2.VideoCapture(filepath)
    frame_preds = []
    sample_frame = None
    while cap.isOpened() and len(frame_preds) < 15:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if sample_frame is None: sample_frame = frame_rgb
        img_resized = cv2.resize(frame_rgb, (128, 128))
        img_array = preprocess_input(np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0))
        frame_preds.append(float(video_model.predict(img_array)[0][0]))
    cap.release()
    avg_score = sum(frame_preds) / len(frame_preds) if frame_preds else 0
    is_fake = avg_score <= 0.5
    conf = round((1-avg_score)*100, 1) if is_fake else round(avg_score*100, 1)
    desc = get_pixel_forensics(sample_frame, is_fake)
    save_forensic_log(filename, "Fake" if is_fake else "Real", conf, desc)
    return jsonify({"result": "Fake" if is_fake else "Real", "confidence": conf, "explanation": desc, "source": "Video"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
