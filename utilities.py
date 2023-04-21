import nltk
import numpy as np
# nltk.download('punkt') *this is the tokenizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() #this creates the stemmer

#tokenizes sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(words) for words in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, words in enumerate(all_words):
        if words in tokenized_sentence:
            bag[i] = 1.0
    return bag
