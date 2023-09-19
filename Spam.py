import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create a sample dataset with 10 messages and labels
data = pd.DataFrame({
    'message': [
        "Free entry to win a prize!",
        "Hey, how are you?",
        "Claim your gift card now!",
        "Meeting at 3 pm?",
        "You have won $1000 cash prize!",
        "Can we reschedule our meeting?",
        "Get a discount on our products today!",
        "Please send me the report.",
        "Congratulations! You've won a vacation.",
        "Are you available for a call?"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['message'])
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
