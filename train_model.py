import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load our dataset
data = pd.read_csv("spam.csv")

# Separate messages and labels
X = data["message"]
y = data["label"]

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train the AI model
model = MultinomialNB()
model.fit(X_train, y_train)

# Check model performance
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("AI Spam Detector trained successfully!")
print(f"Test accuracy: {accuracy:.2%}")

# Test a new message
message = input("Enter a message to check: ")
message_vector = vectorizer.transform([message])
result = model.predict(message_vector)[0]

if result == "spam":
    print("Prediction: SPAM MESSAGE")
else:
    print("Prediction: NORMAL MESSAGE")