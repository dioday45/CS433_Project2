import numpy as np
import string
import re
import nltk
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
nltk.download('words')

UNWANTED_WORDS = ['user', 'url', 'rt']

USEFUL_STOPWORDS = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

words = set(nltk.corpus.words.words())
tweet_tokenizer = nltk.tokenize.TweetTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stopword = set(nltk.corpus.stopwords.words('english'))
for w in USEFUL_STOPWORDS:
    stopword.discard(w)


def remove_stopwords(text):
    """Remove the stopwords from a given string

    Args:
        stopwords (list(string)): words to remove
        text (string): string from whom to remove the words

    Returns:
        string: the resulting string
    """
    return " ".join([word for word in text.split() if word.lower() not in stopword])


def remove_punct(text):
    """Remove the ponctuations from a given string "text" and return the result

    Args:
        text (string): text from whom to remove the punctuations

    Returns:
        string: text without punctuations
    """
    text = re.sub("/", " ", text)  ## First change all '/' to ' '
    return text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )  # Replace all punctuations by ' '


def add_space(text):  # TODO
    """Remove camel cases from text

    Args:
        text (string): text to remove camel cases

    Returns:
        string: text without camel case
    """
    return re.sub("([a-z])([A-Z])", r"\1 \2", text)


def remove_white_space(text):
    """Remove extra white spaces

    Args:
        text (string): text to remove extra white spaces

    Returns:
        string: text without extra white spaces
    """
    return re.sub(" +", " ", text)


def remove_words_digits(text):
    """Remove words containing digits

    Args:
        text (string): text to remove word containing digits

    Returns:
        string: text without camel case
    """
    return re.sub(r"\w*\d\w*", "", text)


def to_lower(text):
    """Change all words to lower cases

    Args:
        text (string): text to modify

    Returns:
        string: text with only lower case words
    """
    return text.lower()

def remove_specific_words(text):
    """Remove URL

    Args:
        text (string): text to modify

    Returns:
        string: text without URL
    """
    return " ".join([word for word in text.split() if word.lower() not in UNWANTED_WORDS])

def remove_repeating_char(text):
    """Remove repeating character

    Args:
        text (string): text to modify

    Returns:
        string: text without repeated character
    """
    return re.sub(r'(.)1+', r'\1\1', text)

def remove_single_char(text):
    """Remove single character word

    Args:
        text (string): text to modify

    Returns:
        string: text without single character word
    """
    return ' '.join( [w for w in text.split() if len(w)>1] )

def lemmatize(text):
    """Lemmatize the text

    Args:
        text (string): text to lemmatize

    Returns:
        string: text lemmatized
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def tokenize(text):
    """Tokenize text

    Args:
        text (str): text to tokenize

    Returns:
        String list: tokenized text
    """

    return tweet_tokenizer.tokenize(text)

def remove_non_english_words(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())


def preprocess_data(df):
    df['tweet'] = df['tweet'].progress_apply(lambda x: to_lower(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_stopwords(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_punct(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: add_space(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_white_space(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_words_digits(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_specific_words(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_repeating_char(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_single_char(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: remove_non_english_words(x))
    df['tweet'] = df['tweet'].progress_apply(lambda x: lemmatize(x))

    df = df[df['tweet'] != '']
    df = df.drop_duplicates()
    df.reset_index(inplace=True)

    return df