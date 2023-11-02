
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Text containing clinical entities
text = "The patient was diagnosed with pneumonia in Room 302."

# Analyze named entities
doc = nlp(text)

# Extract and label named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
