
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("saved_models/random_forest_model.pkl")
tfidf = joblib.load("saved_models/tfidf_vectorizer.pkl")


# Function to predict news
def predict_news(title, text):

    # Check whether both inputs are empty
    if not title.strip() and not text.strip():
        return "Please enter a news article."

    # Combine title and article text
    content = title + " " + text

    # Convert text into TF-IDF format
    news_tfidf = tfidf.transform([content])

    # Make prediction
    prediction = model.predict(news_tfidf)

    # Get prediction probabilities
    probabilities = model.predict_proba(news_tfidf)

    # Display probabilities
    real_probability = probabilities[0][0] * 100
    fake_probability = probabilities[0][1] * 100

    print("\n===== Prediction Probabilities =====")
    print(f"Real News: {real_probability:.2f}%")
    print(f"Fake News: {fake_probability:.2f}%")

    # Return final result
    if prediction[0] == 1:
        return "Fake News"
    else:
        return "Real News"


# Main program
print("===== Fake News Detection System =====")

# Get news title
title = input("\nEnter News Title: ")

# Get multiline news article
print("\nEnter News Article (type END on a new line to finish):")

lines = []

while True:
    line = input()

    if line.strip().upper() == "END":
        break

    lines.append(line)

text = "\n".join(lines)

# Predict news
result = predict_news(title, text)

# Display result
print("\n===== Prediction Result =====")
print("News Prediction:", result)
