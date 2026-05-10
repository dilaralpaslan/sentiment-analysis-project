import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove empty rows
data = data.dropna()

# Features and labels
X = data["Comment"]
y = data["Sentiment"]

# Convert text data into numerical vectors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Sentiment Analysis Project")
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Test custom comment
sample = ["This video was very helpful and easy to understand"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)

print("\nSample Comment:", sample[0])
print("Prediction:", prediction[0])