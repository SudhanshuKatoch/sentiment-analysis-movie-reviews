import streamlit as st
import joblib
import os
from transformers import pipeline
from PIL import Image

# Paths
models_path = r"C:\sentiment-analysis-movie-reviews\models"
banner_path = "data/app_banner.jpg"

# Load Random Forest model and vectorizer
rf_model = joblib.load(os.path.join(models_path, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(models_path, 'vectorizer.pkl'))

# Specify the model for Hugging Face pipeline
hf_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
hf_model = pipeline("sentiment-analysis", model=hf_model_name)

# App configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎥",
    layout="wide",
)

# Sidebar
st.sidebar.title("About the App")
st.sidebar.info(
    "This app analyzes movie reviews and predicts whether the sentiment is Positive or Negative."
    " Choose between a custom-trained model and a pre-trained Hugging Face model for analysis."
)

# Add banner image with error handling
if os.path.exists(banner_path):
    try:
        st.sidebar.image(Image.open(banner_path), use_container_width=True)
    except Exception as e:
        st.sidebar.warning("Could not load the banner image. Error: " + str(e))
else:
    st.sidebar.warning("Banner image not found. Please place 'app_banner.jpg' in the 'data' directory.")

# Main UI
st.title("🎬 Sentiment Analysis on Movie Reviews")
st.markdown(
    "Analyze movie reviews using advanced NLP techniques. Choose between a custom-trained model and a pre-trained model."
)

# User input
st.write("### Enter Your Movie Review")
user_input = st.text_area("Input your review here:", height=150)

# Model selection
st.write("### Choose a Sentiment Analysis Model")
model_choice = st.radio(
    "Select your preferred model:",
    ("Custom-trained Random Forest", "Hugging Face Pre-trained"),
    index=1,  # Set Hugging Face Pre-trained as the default
    help="Choose between a Random Forest model trained on IMDb data or Hugging Face's robust pre-trained model."
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
    "Made with ❤️ using [Streamlit](https://streamlit.io/) | © 2025 Your Name"
)
