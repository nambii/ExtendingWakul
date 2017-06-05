#!/usr/bin/python3
"""detect_capitalisation.py

Program to detect capitalisation in articles
"""
import os
import sys
import numpy as np
import pandas as pd
from csv import DictReader
import couchdb
from config.CouchDBConnect import CouchDBConnect
import config.core as core
import h5py
import pickle
import re
import collections


_db_handle=CouchDBConnect.connect_CouchDB()
db = _db_handle["articles2"]

def get_text(db):
    """Get the id and text field of the articles"""
    queue = {}
    try:
        for row in db.view('article_text/article_text',
                                 wrapper=None,
                                 group='False'
                                 ):
                queue.update({row.key[1]: {}})
                queue[row.key[1]]['text'] = row.key[0]
    except:
        raise Exception("Failed to retrieve view: "
            + db.name
            + "/artciles/_view/article_text\n\n")
    return queue
def iterate_doc(db):
    """Iterate through the doc in databasse"""
    try:
        replies = get_text(db)
        print(replies)
        for r in replies:
            try:
                print(r)
                doc = db.get(r)
                if('text' in doc) :
                   text = doc['text']
                   rwords=[]
                   cnt1=collections.Counter()
                   cnt2=collections.Counter()
                   words = re.findall('\w+', text)
                   #Open the file to read the relevent word list
                   with open('files/words.txt', 'r') as f:
                       lines=[line.rstrip() for line in f]
                       for line in lines:
                           rwords.append(line)
                   #find the words not following the rule
                   for word in words :
                      for rword in rwords :
                        if(word== rword.lower()) :
                            cnt1[rword]+=1
                   #find the words following the rule
                   for word in words :
                      for rword in rwords :
                        if(word== rword) :
                            print("a")
                            cnt2[rword]+=1
                   #find the total number of relevent words
                   counter_value1=0
                   counter_value2=0
                   for key in dict(cnt1):
                       counter_value1=counter_value1+cnt1[key]
                   for key in dict(cnt2):
                       counter_value2=counter_value2+cnt2[key]
                   total=counter_value1+counter_value2
                   #find the percentage of words following the rule
                   percent=0
                   if(total!=0) :
                        percent=(counter_value2)/(counter_value1+counter_value2)*100
                   new_dict = dict(cnt1).copy()
                   new_dict.update(dict(cnt2))
                   #make a dict for storing it with key 'capitalisation'
                   cap=dict(total_count=(counter_value1+counter_value2),percentage=percent,words=dict(upper_case=dict(cnt2),lower_case=dict(cnt1)))
                   doc['capitalisation']=cap
                   #Save the docv
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
  #iterate through the doc in database
  iterate_doc(db)

if __name__ == "__main__":
    main()
