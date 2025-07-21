import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf

# Title
st.title("🎵 Raga Identification App")

# Load model
model = joblib.load("raag_model.pkl")

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp.wav")
        prediction = model.predict([features])[0]
        st.success(f"🎼 Predicted Raga: **{prediction}**")
    except Exception as e:
        st.error(f"Error: {str(e)}")
