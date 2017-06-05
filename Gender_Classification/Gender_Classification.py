#!/usr/bin/python3
"""Gender_Classification

For  classifying the twitter user based gender using name,description or text
"""
import os
import sys
import numpy as np
import pandas as pd
from SentenceTokeniser import SentenceTokeniser
import couchdb
from config.CouchDBConnect import CouchDBConnect
import config.core as core
import h5py
import pickle
import genderizer
import tweepy
from tweepy import OAuthHandler
from tweepy.utils import import_simplejson


_db_handle=CouchDBConnect.connect_CouchDB()
db = _db_handle["users"]
def load_api():
    ''' Function that loads the twitter API after authorizing the user. '''
    args = core.config('twitter','OAuth')

    consumer_key = args['consumer_key']
    consumer_secret = args['consumer_secret']
    access_token = args['access_token']
    access_secret = args['access_secret']
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # load the twitter API via tweepy
    return tweepy.API(auth)
def get_doc(db):
    """Function to get id's and screen_name from users
    """
    queue = {}
    try:
        for row in db.view('users/users_list',
                                 wrapper=None,
                                 group='False'
                                 ):
                queue.update({row.key[0]: {}})
                queue[row.key[0]]['screen_name'] = row.key[1]
                # queue[row.key[0]]['sport'] = row.key[2]
    except:
        raise Exception("Failed to retrieve view: "
            + db.name
            + "/users/_view/users_list\n\n")
    return queue
def iterate_doc(db):
    """Function to  iterate through each tweet to do gender classification for
    user
    """
    json = import_simplejson()

    try:
        # get all users
        docs = get_doc(db)
        for r in docs:
            text=[]
            try:
                doc = db.get(r)
                if('gender' not in doc) :
                    api = load_api()
                    try :
                        # get tweets from user timeline
                        tweets=api.user_timeline(r,count=10)
                        for tweet in tweets:
                                #converting tweet to dict
                                dtweet= json.dumps(tweet._json)
                                #converting tweet in dict to json
                                tweet1= json.loads(dtweet)
                                if('text' in tweet1) :
                                   text.append(tweet1['text'])
                        print(text)
                    except:
                        pass
                    if('description' in doc) :
                       description = doc['description']
                    else :
                       description=None
                    if('name' in doc) :
                       #Retrieve firstname
                       name= doc['name'].split()
                       firstName = name[0]
                       print("firstname : ",firstName,r )
                    else :
                       firstName=None
                    description=doc['description']
                    #check gender
                    gender=genderizer.detect(firstName,text,description)
                    print("Gender:  ",gender)
                    doc['gender']=gender
                    doc = db.save(doc)
                    print ("Saved")
            except:
                print("Exception1")
                pass
    except:
        print("Exception2")
        pass

def main():
  _db_handle=CouchDBConnect.connect_CouchDB()
  db = _db_handle["users"]
  iterate_doc(db)


if __name__ == "__main__":
    main()
