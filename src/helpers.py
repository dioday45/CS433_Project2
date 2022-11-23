import numpy as np
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

UNWANTED_WORDS = ['user', 'url', 'rt']


def remove_stopwords(text):
    """Remove the stopwords from a given string

    Args:
        stopwords (list(string)): words to remove
        text (string): string from whom to remove the words

    Returns:
        string: the resulting string
    """
    return " ".join([word for word in text.split() if word.lower() not in stopwords])


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
    return re.sub(r'(.)1+', r'1', text)

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