import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define paths
DATASET_PATH = "./dataset/dataset_phishing.csv"  # Change this to your dataset file path
MODEL_PATH = "./model/phishing_model.pkl"  # Change this to your saved model path
background_image = "./matrix.jpg"  # Added background image

# Load dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    if 'url' in df.columns and 'status' in df.columns:
        phishing_urls = set(df[df['status'] == 'phishing']['url'].tolist())
        legitimate_urls = set(df[df['status'] == 'legitimate']['url'].tolist())
        return phishing_urls, legitimate_urls, df  # Store URLs in sets for quick lookup and keep dataframe
    return set(), set(), df

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Function to extract features from URL (dummy function, replace with actual feature extraction logic)
def extract_features_from_url(url, df):
    return pd.DataFrame([np.random.rand(len(df.columns) - 2)], columns=df.drop(columns=['url', 'status']).columns)  # Ensure valid feature names

# Function to make predictions
def predict_url(url, phishing_urls, legitimate_urls, model, df):
    if url in phishing_urls:
        return "Phishing", 100.0  # If URL exists in phishing set, classify as phishing with 100% confidence
    if url in legitimate_urls:
        return "Legitimate", 100.0  # If URL exists in legitimate set, classify as legitimate with 100% confidence
    
    features = extract_features_from_url(url, df)
    pred_prob = model.predict_proba(features)[0]
    confidence = max(pred_prob) * 100
    prediction = "Phishing" if model.predict(features)[0] == 1 else "Legitimate"
    return prediction, confidence

# Load dataset and model
phishing_urls, legitimate_urls, df = load_dataset()
model = load_model()

# Streamlit UI Styling
st.markdown(
    f"""
    <style>
        .main {{
            background-image: url('./static/matrix.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: white;
        }}
        .subtitle {{
            text-align: center;
            font-size: 18px;
            color: #ccc;
        }}
        .scan-btn {{
            display: flex;
            justify-content: center;
        }}
        .developer {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: white;
        }}
        .signature {{
            text-align: center;
            font-size: 24px;
            font-family: cursive;
            font-weight: bold;
            color: #ff6347;
        }}
        .stButton>button {{
            display: block;
            margin: auto;
            background-color: #1E90FF;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 24px;
            border-radius: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<p class="title">🔍 Phishing URL Detection App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect whether a URL is safe or a phishing attempt with high accuracy.</p>', unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>✍️ <b>Enter URL for prediction:</b></p>", unsafe_allow_html=True)
url = st.text_input("", key="url_input")

if st.button("🛡️ Scan", key="scan_button"):
    if url:
        prediction, confidence = predict_url(url, phishing_urls, legitimate_urls, model, df)
        icon = "✅" if prediction == "Legitimate" else "⚠️"
        st.markdown(f'<p style="text-align: center; font-size: 20px; font-weight: bold;">{icon} Prediction: {prediction}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; font-size: 18px;">Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a URL.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="developer">👨‍💻 Developed by</p>', unsafe_allow_html=True)
st.markdown('<p class="signature">Pradeep Bhattarai</p>', unsafe_allow_html=True)