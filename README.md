Sentiment Analysis on Movie Reviews 🎮

Overview

This project leverages cutting-edge Natural Language Processing (NLP) techniques to classify movie reviews as positive or negative. It provides flexibility through multiple approaches, including a machine learning model trained on IMDb data and a pre-trained transformer model from Hugging Face. The project features an interactive user interface for real-time predictions and batch processing.

Features

Data Preprocessing:

Utilizes NLTK for text cleaning, tokenization, and stopword removal.

Converts text data into numerical form using TF-IDF vectorization.

Machine Learning Models:

A custom-trained Random Forest Classifier for sentiment prediction.

A robust pre-trained transformer model from Hugging Face (distilbert-base-uncased-finetuned-sst-2-english) for enhanced accuracy.

Model Comparison:

Users can seamlessly switch between the Random Forest model and the Hugging Face transformer for analysis.

Interactive Interface:

Built with Streamlit, offering an intuitive user interface for:

Single review analysis: Input individual movie reviews for quick sentiment analysis.

Batch processing: Upload a file containing multiple reviews to process them in bulk.

GPU/CPU Flexibility:

Automatically detects and utilizes GPU (if available) for faster Hugging Face model predictions. Falls back to CPU when a GPU is not accessible.

Visualization:

Provides sentiment confidence scores and clear visual results.

What's Included

Preprocessing Pipeline:

Cleaning, tokenization, and vectorization of input data.

Custom Machine Learning Model:

Random Forest model trained on IMDb movie reviews for balanced performance.

Transformer-Based Sentiment Analysis:

Pre-trained model from Hugging Face for state-of-the-art results.

Streamlit Web App:

User-friendly interface to input reviews or upload files for analysis.

Supports switching between models in real-time.

Deployment Ready:

Designed for local use and cloud deployment.

How to Use

Run the Application:

streamlit run app.py

Input Options:

Type in individual movie reviews.

Upload a CSV file for batch processing.

Select Model:

Choose between:

Hugging Face Pre-trained Model for enhanced accuracy.

Custom Random Forest Model for a lightweight solution.

View Results:

Get predictions with sentiment labels (Positive/Negative) and confidence scores.

Technologies Used

Programming Language: Python

Libraries/Frameworks:

NLP: NLTK, Transformers

Model Training: Scikit-learn

Web App: Streamlit

Deployment Ready: Docker-compatible

Deployment Options

Local deployment using Streamlit.


