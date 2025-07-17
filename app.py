from flask import Flask, request, jsonify, render_template_string
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

# Home page with file upload
@app.route('/')
def index():
    return render_template_string('''
        <h2>🎵 Upload an Audio File for Raaga Identification</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="file" accept=".wav" required><br><br>
            <input type="submit" value="Identify Raaga">
        </form>
    ''')

# Prediction route
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
        return f"<h3>🎼 Predicted Raaga: {prediction}</h3>"
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
