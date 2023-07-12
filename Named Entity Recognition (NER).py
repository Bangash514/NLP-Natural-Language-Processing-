#Created by Bangash Owais, Software Engineer, Shenzhen China.

"""
import nltk

#Note: If you did not download the below resuceses then must do it.  
#nltk.download('punkt')  # Download necessary resources for tokenization
#nltk.download('averaged_perceptron_tagger')  # Download necessary resources for POS tagging
#nltk.download('maxent_ne_chunker')  # Download necessary resources for NER
#nltk.download('words')  # Download necessary resources for NER

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Input text
text = "I Love China, rich culture and having modren architures. Pakistan is best friend of China!."

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
pos_tags = pos_tag(tokens)

# Perform NER
ner_tags = ne_chunk(pos_tags)

# Print the named entities
for entity in ner_tags:
    if hasattr(entity, 'label'):
        print(entity.label(), ' '.join(c[0] for c in entity))
