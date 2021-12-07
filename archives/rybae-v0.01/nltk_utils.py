import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentece, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [     0,      1,      0,      1,   0,       0,     0, ]
    """
    tokenized_sentece = [stem(w) for w in tokenized_sentece]
    # create an array of zeroes, same length as all_words[]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentece:
            bag[index] = 1.0

    return bag
