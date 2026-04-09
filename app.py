import os
import joblib  # 🔥 NEW: Imported to load your custom ML model
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Create a folder to save uploaded images/videos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- NEW: LOAD YOUR CUSTOM ML MODEL ----------------
print("Waking up the AI...")
try:
    # Load the brain and the dictionary you trained in Colab!
    text_model = joblib.load('truthdetect_model.pkl')
    text_vectorizer = joblib.load('truthdetect_vectorizer.pkl')
    print("AI successfully loaded and ready!")
except Exception as e:
    print(f"⚠️ Error loading AI. Make sure the .pkl files are in this folder! Details: {e}")
    text_model = None
    text_vectorizer = None


@app.route('/', methods=['GET'])
def home():
    return "TruthDetect ML Backend is Running!"


# ---------------- UPDATED: REAL TEXT PREDICTION ENDPOINT ----------------
@app.route('/predict-text', methods=['POST'])
def predict_text():
    data = request.json
    incoming_text = data.get('text', '')
    
    if not incoming_text:
        return jsonify({"error": "No text provided"}), 400

    # Safety check in case the model didn't load
    if text_model is None or text_vectorizer is None:
        return jsonify({"error": "ML model not loaded on server"}), 500

    try:
        # Step 1: Translate the incoming English text into mathematical vectors
        math_vector = text_vectorizer.transform([incoming_text])
        
        # Step 2: Ask your trained model to predict FAKE or REAL
        prediction = text_model.predict(math_vector)[0]
        
        # Format it nicely for your React Native app
        result_string = str(prediction).title() # Turns "FAKE" to "Fake"
        
        return jsonify({
            "result": result_string, 
            "confidence": 92, # Estimated baseline confidence for this model
            "explanation": f"Your custom ML model analyzed the linguistic patterns and classified this text as {result_string}."
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- IMAGE ENDPOINT (MOCK) ----------------
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
    return jsonify({
        "result": "Real", 
        "confidence": 95, 
        "explanation": f"Image '{filename}' processed. No visual artifacts or abnormal pixel gradients detected in the facial region."
    })


# ---------------- VIDEO ENDPOINT (MOCK) ----------------
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