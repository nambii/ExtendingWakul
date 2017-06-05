"""user_saerch.py
To search Indigenous Auatralian users on twitter
"""
import tweepy
from tweepy import OAuthHandler
import json
import datetime as dt
import time
import os
import sys
import couchdb
from tweepy.utils import import_simplejson
from config.CouchDBConnect import CouchDBConnect
import config.core as core



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


def user_search(api,query,maxi=20,page=1) :
    """Funtion to search user using tweepy user search API"""
    searched_users = []

    while(len(searched_users) < maxi) :
        remaining_users = maxi - len(searched_users)
        try:
            new_users = api.search_users(query,maxi,page)
            print('found', len(new_users), 'users')
            if not new_users:
                print('no users found')
                break
            searched_users.extend(new_users)
        except tweepy.TweepError:
            print('exception raised, waiting 15 minutes')
            print('(until:', dt.datetime.now() + dt.timedelta(minutes=15), ')')
            time.sleep(15 * 60)
            break  # stop the loop
    return searched_users


def write_users(users,query):
    ''' Function that appends users to couchdb. '''

    json = import_simplejson()
    _db_handle=CouchDBConnect.connect_CouchDB()
    db = _db_handle["users"]
    duser={}
    for user in users:

            duser= json.dumps(user._json)
            store_user(db,duser,query)


def store_user(db,duser,query) :
    """User as Dict"""
    query=query.lower()
    try:
        flag=False
        doc = json.loads(duser)
        doc.update({'_id': doc['id_str']})
        description=doc['description'].lower()
        n=doc['name'].lower()
        if(description.find(query)!=-1) :
            if(checkLocation(doc['time_zone'],doc['location'])) :
               flag=True
            if(doc['lang']!=None) :
                if(doc['lang']!="en" or doc['status']['lang']!="en") :
                    flag = False
        elif(len(n)>0) :
           name=n.split()
           if(name[0].find(query)!=-1) :
              if(checkLocation(doc['time_zone'],doc['location'])) :
                 print(doc['time_zone'],"  ",doc['location'])
                 flag=True

        # doc.update({'team_name' : 'Adelaide Crows'})
    except KeyError:
        print(
            "Warning: store_user() called, but user input "
            + "parameter does not contain an id_str."
        )
        return None
    if not isinstance(doc, dict):
        print(
            "Warning: store_dict() called, but doc input parameter"
            + " is not a dict."
        )
        return None
    # Attempt to save document to CouchDB
    try:
        if(flag) :
          print("Stored")
          response = db.save(doc)
    # If the _id already exists
    except couchdb.http.ResourceConflict as e:
        print(
            "Warning: The doc (_id: "
            + doc['_id']
            + ") is already exists "
        )

        print(
            "Overwriting doc (_id: "
            + doc['_id']
            + ")"
        )

        # #Delete the document from the database(if required)
        # for r in db.revisions(doc['_id']):
        #     db.purge([{'_id': doc['_id'], '_rev': r.rev}])
        #
        # response = store_tweet(db,dtweet)
        # return response
def checkLocation(time_zone,location) :
    """Function to check the location and time_zone of the user"""
    print("Checking Location")
    flag=False
    cities=[]
    with open("files/cities.txt","r") as f:
        lines=[line.rstrip() for line in f]
        for line in lines:
            cities.append(line)
    print(cities)
    for city in cities :
        if(len(city)>1) :
            if(time_zone!=None) :
               if(time_zone.find(city) != -1) :
                  print(city)
                  flag=True
            elif(location!=None) :
               if(location.find(city) != -1) :
                  print(city)
                  flag=True
    return flag



def main():
    ''' This is a script that continuously searches for users
        based on search phrase.'''
    search_phrases=[]
    with open("files/names.txt","r") as f:
        lines=[line.rstrip() for line in f]
        for line in lines:
            search_phrases.append(line)
    # print(search_phrases)

    api = load_api()

    if(True):
        for search_phrase in search_phrases:
            print("Now searching : ",search_phrase)
            exitcount=1
            while(exitcount<3):
                # collect users
                users=user_search(api,search_phrase,20,exitcount)
                # write user details to couchdb in JSON format
                if users:
                    write_users(users,search_phrase)
                    print("Entering")
                    with open("files/users.json", 'a') as f:
                        for user in users:
                            json.dump(user._json, f)
                            f.write('\n')
                exitcount +=1
                print(exitcount)
                if exitcount == 3:
                    if search_phrase == search_phrases[-1]:
                        sys.exit('Maximum number of empty user strings reached - exiting')
                    else:
                        print('Maximum number of empty user strings reached - breaking')
                        break


if __name__ == "__main__":
    main()
