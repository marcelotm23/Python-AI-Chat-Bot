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
#Stammer, take each word in our pattern and bring it down to the root word
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)# return a list of words
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])