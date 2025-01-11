import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import joblib
from tqdm import tqdm
from PIL import Image
import streamlit as st

# Ensure necessary NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Create the 'data' directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Create a blank image (RGB mode) with dimensions 800x200
image = Image.new('RGB', (800, 200), color = (73, 109, 137))

# Save the image in the 'data' directory
image_path = 'data/app_banner.jpg'
image.save(image_path)
print("Image created successfully!")

# Load dataset
print("Loading dataset...")
dataset = pd.read_csv('data/IMDB Dataset.csv')
print(dataset.head())

# Use a subset of 30,000 rows for training
dataset = dataset.head(30000)  # Limit to 30,000 rows
print(f"Dataset reduced to {len(dataset)} rows.")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Enable progress bar for preprocessing
tqdm.pandas()
print("Preprocessing started...")
dataset['cleaned_reviews'] = dataset['review'].progress_apply(preprocess_text)
print("Preprocessing completed.")

# Vectorize the text data
print("Vectorization started...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['cleaned_reviews']).toarray()
print("Vectorization completed.")

# Convert sentiment to binary (1 for positive, 0 for negative)
y = dataset['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the dataset into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split completed.")

# Train the model
print("Model training started...")
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the model and vectorizer
models_path = r"C:\sentiment-analysis-movie-reviews\models"
os.makedirs(models_path, exist_ok=True)

print("Saving model and vectorizer...")
joblib.dump(model, os.path.join(models_path, 'sentiment_model.pkl'))
joblib.dump(vectorizer, os.path.join(models_path, 'vectorizer.pkl'))
print(f"Model and vectorizer saved successfully in {models_path}.")

# Display the image in Streamlit
st.sidebar.image(Image.open(image_path), use_column_width=True)
