#File Created by; Bangash Owais

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Input text and corresponding labels
texts = [
    "I love this movie! It's fantastic.",
    "This book is boring.",
    "The restaurant service was terrible.",
    "The concert was amazing!"
]
labels = ['positive', 'negative', 'negative', 'positive']

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Perform sentiment analysis on new texts
new_texts = [
    "I had a great time at the party!",
    "The weather is awful today."
]
new_X = vectorizer.transform(new_texts)
predictions = classifier.predict(new_X)

# Print the predictions
for text, prediction in zip(new_texts, predictions):
    print(f"Text: {text}\nSentiment: {prediction}\n")
