import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentece, all_words):
    pass


a = "how long does shipping take?"
print(a)
a = tokenize(a)
print(a)

words = ['organize', 'organizer', 'organizing']
stemmed_word = [stem(word) for word in words]
print(stemmed_word)
