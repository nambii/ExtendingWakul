"""SentenceTokeniser.py
To preprocess text data
"""
import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


class SentenceTokeniser(object):

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        lemma = WordNetLemmatizer()
        review_text = BeautifulSoup(review,"lxml").get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them

        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            operators = set(('and', 'or', 'not'))
            stop = stops-operators
            normalised = [w for w in words if not w in stop]
        return(normalised)
