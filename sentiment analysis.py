#Created by; Owais Bangash, Shenzhen China

from textblob import TextBlob

# Input text
text = "I love this movie! It's fantastic."

# Perform sentiment analysis
blob = TextBlob(text)
sentiment = blob.sentiment

# Print the sentiment polarity and subjectivity
print("Polarity:", sentiment.polarity)  # Polarity ranges from -1 (negative) to 1 (positive)
print("Subjectivity:", sentiment.subjectivity)  # Subjectivity ranges from 0 (objective) to 1 (subjective)
