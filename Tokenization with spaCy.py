The simplified code examples using Python and common NLP libraries like spaCy and NLTK. These examples showcase basic text processing and entity recognition, which are often used in NLP for healthcare applications:
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Text to be tokenized
text = "The patient presented with chest pain and shortness of breath."

# Tokenize the text
doc = nlp(text)

# Print the tokens
for token in doc:
    print(token.text)
