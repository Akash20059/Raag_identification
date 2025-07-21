import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf

# Load model
model = joblib.load("raag_model.pkl")

# Extract features
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfccs.T, axis=0)

# UI
st.title("🎵 Raaga Identification App")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp.wav")
        prediction = model.predict([features])[0]
        st.success(f"🎼 Predicted Raaga: {prediction}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
