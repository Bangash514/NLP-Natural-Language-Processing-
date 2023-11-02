
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Text with patient feedback
text = "The hospital staff was very caring and helpful. However, the wait times were quite long."

# Analyze sentiment
sentiment = analyzer.polarity_scores(text)

# Print sentiment scores
print("Sentiment Analysis Results:")
print("Positive:", sentiment['pos'])
print("Negative:", sentiment['neg'])
print("Neutral:", sentiment['neu'])
print("Compound:", sentiment['compound'])
