#!/usr/bin/python3
"""Topic_Modelling.py

Iterate through each article and find topics.
"""
import os
import sys
import numpy as np
import pandas as pd
from SentenceTokeniser import SentenceTokeniser
from csv import DictReader
import couchdb
from config.CouchDBConnect import CouchDBConnect
import config.core
import h5py
import pickle
import topic as T



def get_text(db):
    """Function to get article text from database
    """
    queue = {}
    try:
        for row in db.view('article_text/article_text',
                                 wrapper=None,
                                 group='False'
                                 ):
                queue.update({row.key[1]: {}})
                queue[row.key[1]]['text'] = row.key[0]
                # queue[row.key[0]]['text'] = row.key[2]
    except:
        raise Exception("Failed to retrieve view: "
            + db.name
            + "/article_text/_view/article_text\n\n")
    return queue
def iterate_doc(db):
    """Function to iterate through all articles in database
    """
    try:
        replies = get_text(db)
        print(replies)
        for r in replies:
            try:
                print(r)
                #check whether text field is present
                doc = db.get(r)
                if('text' in doc) :
                    text = doc['text']
                    if('meta' in doc) :
                       doc1 = doc['meta']
                       if('title' in doc1) :
                          title=doc1['title']
                       else :
                          title='A'
                    else :
                       title='A'
                    #check whether topics are present
                    if('topics' in doc) :
                       print("topics present")
                    else :
                       text = doc['text']
                       #find the topics
                       topics=T.findTopics(text,title)
                       i=0
                       #combine all topics
                       for topic in topics :
                         doc.update({'topics': {}})
                         doc['topics'][str(i)]=topic
                         i=i+1
                       #save the document with topics field back to database
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
