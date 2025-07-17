from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
import soundfile as sf

app = Flask(__name__)

# Load the trained model
model = joblib.load("raag_model.pkl")

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

@app.route('/')
def home():
    return "🎵 Raga Identification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    try:
        features = extract_features(file_path)
        prediction = model.predict([features])[0]
        return jsonify({'predicted_raga': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
