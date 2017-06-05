#!/usr/bin/python3
"""Sentiment_twitter.py

Iterate through each tweet and find the sentiment for each tweet.
"""
import os
import sys
import numpy as np
import pandas as pd
from SentenceTokeniser import SentenceTokeniser
from csv import DictReader
import couchdb
from config.CouchDBConnect import CouchDBConnect
import config.core as core
import h5py
import pickle
import Sentiment_glove as S


def get_text(db):
    queue = {}
    try:
        for row in db.view('tweet_text/tweet_text',
                                 wrapper=None,
                                 group='False'
                                 ):
                queue.update({row.key[1]: {}})
                queue[row.key[1]]['text'] = row.key[0]
                # queue[row.key[0]]['text'] = row.key[2]
    except:
        raise Exception("Failed to retrieve view: "
            + db.name
            + "/tweets/_view/tweet_text\n\n")
    return queue
def iterate_doc(db):
    try:
        replies = get_text(db)
        for r in replies:
            try:
                print(r)
                doc = db.get(r)
                if('text' in doc) :
                    text = doc['text']
                    if('sentiment' in doc) :
                       print("sentiment present")
                    else :
                       text = doc['text']
                       sentiment1=S.SentimentCheck(text)
                       if(sentiment1[0][1]>0.7) :
                          s="positive"
                       elif (sentiment1[0][0]>0.7) :
                          s="negative"
                       else :
                          s="neutral"
                       #set the sentiment field in doc
                       doc['sentiment']=s
                       #save the tweet document in database
                       doc = db.save(doc)
                       print ("Saved")
            except:
                print("Exception1")
                pass
    except:
        print("Exception2")
        pass
def main():
  print
  _db_handle=CouchDBConnect.connect_CouchDB()
  db = _db_handle["articles2"]
  iterate_doc(db)

if __name__ == "__main__":
    main()
