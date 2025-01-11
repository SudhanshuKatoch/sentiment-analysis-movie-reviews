from transformers import pipeline

# Load a pre-trained sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

# Predict sentiment
review = "This movie was fantastic! I loved it."
result = sentiment_model(review)
print(result)
