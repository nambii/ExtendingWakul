"""SentenceTokeniser.py
To  tag part of speech and preprocess text data
"""
import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


class SentenceTokeniser(object):
    """SentenceTokeniser is a utility class for processing  text """

    @staticmethod
    def review_to_wordlist( review, remove_stopwords,part ):
        """ Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        """
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
        # 4. To lemmatize the text data
        normalized = " ".join(lemma.lemmatize(word) for word in words)
        #
        # 5. Tag each word with part of speech tags
        tokens = nltk.word_tokenize(normalized)
        tags=nltk.pos_tag(tokens)
        #
        # 6. Consider only adjetives,nouns and verbs
        dt_tags = " ".join(t[0] for t in tags if t[1] == "JJ" or t[1] == "NN" or t[1] == "VB" and len(t[0])>1)
        words = dt_tags.lower().split()
        #
        # 7. Remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            if(part=='t') :
                extra = set('not')
            else :
                extra = set(('aboriginal', 'indigenous', 'australia','australian','share'))
            words = " ".join(w for w in words if not w in stops and not w in extra )
        return(words)
