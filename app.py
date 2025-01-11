import streamlit as st
import joblib
import os
from transformers import pipeline
from PIL import Image
import torch  # To check for GPU availability

# Paths
models_path = r"C:\sentiment-analysis-movie-reviews\models"

# Load Random Forest model and vectorizer
rf_model = joblib.load(os.path.join(models_path, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(models_path, 'vectorizer.pkl'))

# Detect device for Hugging Face model
device = 0 if torch.cuda.is_available() else -1

# Load Hugging Face sentiment analysis pipeline
hf_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",
    device=device
)

# App configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📚",
    layout="wide",
)

# Sidebar
st.sidebar.title("About the App")
st.sidebar.info(
    "This app analyzes movie reviews and predicts whether the sentiment is Positive or Negative."
    " Choose between a custom-trained model and a pre-trained Hugging Face model for analysis."
)

try:
    st.sidebar.image(Image.open("data/app_banner.jpg"), use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Banner image not found. Please check the path.")

# Main UI
st.title("🎞️ Sentiment Analysis on Movie Reviews")
st.markdown(
    "Use this app to analyze the sentiment of a movie review using cutting-edge NLP techniques."
)

# User input
st.write("### Enter Your Movie Review")
user_input = st.text_area("Input your review here:", height=150)

# Model selection
st.write("### Choose a Sentiment Analysis Model")
model_choice = st.radio(
    "Select your preferred model:",
    ("Hugging Face Pre-trained", "Custom-trained Random Forest"),
    index=0,  # Default selection set to Hugging Face
    help="Choose between a Random Forest model trained on a subset of IMDb data or Hugging Face's robust pre-trained model."
)

# Analyze sentiment
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        if model_choice == "Custom-trained Random Forest":
            # Preprocess and predict using Random Forest model
            input_vectorized = vectorizer.transform([user_input]).toarray()
            prediction = rf_model.predict(input_vectorized)
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
            st.success(f"**Sentiment (Random Forest):** {sentiment}")
        elif model_choice == "Hugging Face Pre-trained":
            # Predict using Hugging Face model
            result = hf_model(user_input)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            st.success(f"**Sentiment (Hugging Face):** {sentiment} \n\n **Confidence:** {confidence:.2f}")
    else:
        st.error("Please enter a valid review.")

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using [Streamlit](https://streamlit.io/) | © 2025 Sudhanshu Katoch"
)
