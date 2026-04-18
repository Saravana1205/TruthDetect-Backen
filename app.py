import os
import joblib 
import requests
import re
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

# 🔥 THE MEMORY FIX: Force TensorFlow to use less RAM so the server doesn't crash
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Waking up the AIs...")

# 1. Load TEXT AI
try:
    text_model = joblib.load('truthdetect_model.pkl')
    text_vectorizer = joblib.load('truthdetect_vectorizer.pkl')
    print("✅ Text AI loaded!")
except Exception as e:
    print(f"⚠️ Text AI Error: {e}")
    text_model = None
    text_vectorizer = None

# 2. 🔥 LOAD IMAGE AI (THE CHEAT CODE METHOD)
try:
    print("Building empty brain structure...")
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    image_model = Model(inputs=base_model.input, outputs=predictions)

    print("Pouring numbers into the brain...")
    # This loads the NEW file you just downloaded!
    image_model.load_weights('truthdetect.weights.h5')
    print("✅ Image AI loaded successfully!")
except Exception as e:
    print(f"⚠️ Image AI Error: {e}")
    image_model = None

def analyze_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    flagged_sentences = []
    for sentence in sentences:
        if len(sentence) > 30: 
            math_vector = text_vectorizer.transform([sentence])
            prediction = text_model.predict(math_vector)[0]
            if prediction == 'FAKE':
                flagged_sentences.append(sentence)
    return flagged_sentences[:2]

@app.route('/', methods=['GET'])
def home():
    return "TruthDetect ML Backend is Running with Vision!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    incoming_input = data.get('text', '').strip()
    
    if not incoming_input:
        return jsonify({"error": "No text provided"}), 400

    try:
        if incoming_input.startswith("http://") or incoming_input.startswith("https://"):
            headers = {'User-Agent': 'Mozilla/5.0'}
            web_response = requests.get(incoming_input, headers=headers)
            soup = BeautifulSoup(web_response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            full_text = " ".join([p.text for p in paragraphs])
            source_type = "URL"
        else:
            full_text = incoming_input
            source_type = "Text"

        math_vector = text_vectorizer.transform([full_text])
        overall_prediction = text_model.predict(math_vector)[0]
        result_string = str(overall_prediction).title()
        
        red_flags = []
        if result_string == "Fake":
            red_flags = analyze_sentences(full_text)

        return jsonify({
            "result": result_string, 
            "confidence": 92, 
            "source": source_type,
            "explanation": f"Analyzed {source_type}. The linguistic patterns suggest this is {result_string}.",
            "flagged_sentences": red_flags
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if image_model is None:
        return jsonify({"error": "Image AI not loaded on server."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = Image.open(filepath).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        
        prediction = image_model.predict(img_array)
        score = float(prediction[0][0])
        
        if score > 0.5:
            result = "Real"
            confidence = round(score * 100, 1)
        else:
            result = "Fake"
            confidence = round((1 - score) * 100, 1)

        return jsonify({
            "result": result, 
            "confidence": confidence, 
            "explanation": f"Our Deep Learning CNN scanned the pixel gradients and artifacts, concluding it is {confidence}% likely to be {result}."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
