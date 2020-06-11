import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

import json

# Load data to train the model
with open('intents.json') as file:
    data = json.load(file)

# Extract data
words = []
labels = []
docs_x = []
docs_y = []
# Stammer, take each word in our pattern and bring it down to the root word
# to reduce the vocabulary of our model and attempt to find the more general
# meaning behind sentences.
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)# return a list of words
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Lower words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
# Sort words
words = sorted(list(set(words)))
# Sort labels
labels = sorted(labels)


# Training, look if the root word of different words from patterns are exist
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Transform input to numpy
training = numpy.array(training)
output = numpy.array(output)