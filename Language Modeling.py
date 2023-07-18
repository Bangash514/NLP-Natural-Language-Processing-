import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Input text
text = """The cat sat on the mat. The dog barked. The cat meowed."""

# Tokenize the text and generate input-output pairs
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

input_sequences = []
output_sequences = []
for i in range(1, len(sequences)):
    input_sequences.append(sequences[:i])
    output_sequences.append(sequences[i])

# Pad sequences to have the same length
max_len = max(len(seq) for seq in input_sequences)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len)

# Convert output sequences to one-hot vectors
num_classes = vocab_size
output_sequences = np.array(output_sequences)
one_hot_output = np.zeros((output_sequences.shape[0], num_classes))
one_hot_output[np.arange(output_sequences.shape[0]), output_sequences - 1] = 1

# Create and train the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, vocab_size)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(padded_input_sequences, one_hot_output, epochs=100, verbose=0)

# Generate new text based on the learned language model
seed_text = "The cat"
seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
for _ in range(10):
    padded_seed = pad_sequences([seed_sequence], maxlen=max_len)
    predicted_index = np.argmax(model.predict(padded_seed), axis=-1)[0]
    predicted_word = next(word for word, index in tokenizer.word_index.items() if index == predicted_index)
    seed_text += " " + predicted_word
    seed_sequence.append(predicted_index)

print("Generated Text:", seed_text)
