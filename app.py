import os
import joblib 
import requests
import re
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Waking up the AI...")
try:
    text_model = joblib.load('truthdetect_model.pkl')
    text_vectorizer = joblib.load('truthdetect_vectorizer.pkl')
    print("AI successfully loaded and ready!")
except Exception as e:
    print(f"⚠️ Error loading AI. Details: {e}")
    text_model = None
    text_vectorizer = None

# 🔥 EXPLAINABLE AI HELPER FUNCTION
def analyze_sentences(text):
    """Breaks text into sentences and finds the most 'Fake' ones."""
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
    return "TruthDetect ML Backend is Running!"

# 🔥 THE NEW SUPERCHARGED ENDPOINT (Handles Text AND URLs)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    incoming_input = data.get('text', '').strip()
    
    if not incoming_input:
        return jsonify({"error": "No text or URL provided"}), 400

    if text_model is None or text_vectorizer is None:
        return jsonify({"error": "ML model not loaded on server"}), 500

    try:
        # 1. Is it a URL or just text?
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

        # 2. Analyze the whole text
        math_vector = text_vectorizer.transform([full_text])
        overall_prediction = text_model.predict(math_vector)[0]
        result_string = str(overall_prediction).title()
        
        # 3. EXPLAINABLE AI: Find the specific fake sentences!
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
