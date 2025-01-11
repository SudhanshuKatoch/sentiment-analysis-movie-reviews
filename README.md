🎬 Sentiment Analysis on Movie Reviews 🎮
Overview
Dive into the world of movie reviews with our state-of-the-art Sentiment Analysis project. Using advanced Natural Language Processing (NLP) techniques, this project classifies movie reviews as positive or negative, offering flexibility through multiple approaches. Whether you're looking for real-time predictions or batch processing, this interactive platform has you covered.

Features
Data Preprocessing
Text Cleaning & Tokenization: Powered by NLTK for spotless data.

TF-IDF Vectorization: Converts text into numerical form for analysis.

Machine Learning Models
Random Forest Classifier: Custom-trained on IMDb data for sentiment prediction.

Hugging Face Transformer: Utilizes distilbert-base-uncased-finetuned-sst-2-english for top-notch accuracy.

Model Comparison
Seamless Switching: Easily switch between Random Forest and Hugging Face models for analysis.

Interactive Interface
Streamlit Web App: Intuitive UI for:

Single Review Analysis: Quick sentiment analysis of individual reviews.

Batch Processing: Upload and process multiple reviews in bulk.

GPU/CPU Flexibility
Automatic Detection: Utilizes GPU if available, otherwise falls back to CPU for predictions.

Visualization
Confidence Scores: Clear visual representation of sentiment confidence.

What's Included
Preprocessing Pipeline
Data Cleaning & Tokenization

TF-IDF Vectorization

Custom Machine Learning Model
Random Forest: Trained on IMDb reviews for balanced performance.

Transformer-Based Sentiment Analysis
Pre-trained Model: From Hugging Face for superior results.

Streamlit Web App
User-Friendly Interface: Input reviews or upload files seamlessly.

Real-Time Model Switching

Deployment Ready
Local & Cloud Deployment: Designed for flexibility.

How to Use
Run the Application
bash
streamlit run app.py
Input Options
Type Individual Reviews: Quick sentiment analysis.

Upload CSV: Batch process multiple reviews.

Select Model
Hugging Face Pre-trained Model: Enhanced accuracy.

Custom Random Forest Model: Lightweight solution.

View Results
Sentiment Labels: Positive/Negative with confidence scores.

Technologies Used
Programming Language: Python

Libraries/Frameworks:

NLP: NLTK, Transformers

Model Training: Scikit-learn

Web App: Streamlit

Deployment Ready: Docker-compatible

Deployment Options
Local Deployment: Using Streamlit.
