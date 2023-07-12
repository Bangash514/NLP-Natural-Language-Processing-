#Import the required libraries
import nltk

#Select the folder where the ,txt file is saved
text_file = open("Downloads/NLP.txt")

#Read the file
text = text_file.read()
print(type(text))
print("\n")

#Print the text
print(text)
print("\n")

#print the lens
print (len(text))
