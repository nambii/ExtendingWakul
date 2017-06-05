"""gender.py

Function to find the gender through description and text
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from SentenceTokeniser import SentenceTokeniser
from csv import DictReader
from pathlib import Path
import h5py
import pickle
import re

def checkDescription(text) :
    """Function to  find the gender based on the description"""


    labels=[]
    train=[]
    des=[]
    file = Path("files/gender.pickle")
    if(not file.exists()) :
        # Retrieve  text and labels for training
        with open("gender-classifier-DFE-791531.csv",encoding="latin-1") as f:
                    for row in DictReader(f):
                        label= row["gender"]
                        labels.append([label])
                        train.append(row["description"])
        clean_description = []
        for i in range( 0, len(train)):
                clean_description.append(" ".join(SentenceTokeniser.review_to_wordlist(train[i], True)))
                print(i)
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(labels)
        with open('files/mlb.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump(mlb, f)

        print(labels)
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('clf', OneVsRestClassifier(LinearSVC()))])
        classifier.fit(clean_description, Y)
        # Save the classifier as pickle
        with open('files/gender.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump(classifier, f)

    des.append(" ".join(SentenceTokeniser.review_to_wordlist(text, True)))
    # Load the classifier from pickle file
    with open('files/gender.pickle','rb') as f:  # Python 3: open(..., 'rb')
           classifier = pickle.load(f)
    predicted = classifier.predict(des)
    with open('files/mlb.pickle','rb') as f:  # Python 3: open(..., 'rb')
           mlb = pickle.load(f)
    all_labels = mlb.inverse_transform(predicted)
    try :
       return(all_labels[0][0])
    except :
       return None

def checkTweet(texts) :
    """Function to  find the gender bracket into which most tweets were classified"""
    lst=[0,0,0,0]
    for text in texts :
        g=getGenderfortext(text)
        if (g== 'male') :
            lst[0]=lst[0]+1
        elif (g== 'female') :
            lst[1]=lst[1]+1
        elif (g== 'brand') :
            lst[2]=lst[2]+1
        else :
            lst[3]=lst[3]+1
    print('male :',lst[0])
    print('female :',lst[1])
    print('brand :',lst[2])
    print('None :',lst[3])
    _max=max(lst)
    ans=np.array(lst)
    ans1 = np.where( ans==_max )
    # ans=ans.index(_max)
    print(ans1[0])
    a=ans1
    ans=ans1[0].flatten()
    print("ans : ",ans)
    if(len(ans)>1) :
        if('3' in str(ans)) :
            ans=[ans[0]]

    if(len(ans)==1) :
        if ans[0]==0 :
            return 'male'
        elif ans[0]==1 :
            return 'female'
        elif ans[0]==2 :
            return 'brand'
        else :
            return None
    else :
        return 'human'




def getGenderfortext(text) :
    """Function to find the gender from tweet"""
    labels=[]
    train=[]
    des=[]
    file = Path("files/gendertext.pickle")
    if(not file.exists()) :
        # Retrieve  text and labels for training
        with open("gender-classifier-DFE-791531.csv",encoding="latin-1") as f:
                    for row in DictReader(f):
                        label= row["gender"]
                        labels.append([label])
                        train.append(row["text"])
        clean_text = []
        for i in range( 0, len(train)):
                clean_text.append(" ".join(SentenceTokeniser.review_to_wordlist(train[i], True)))
                print(i)
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(labels)
        with open('files/mlb.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump(mlb, f)

        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('clf', OneVsRestClassifier(LinearSVC()))])
        # Fit the data using the classifier
        classifier.fit(clean_text, Y)
        #  Save classifer as pickle file
        with open('files/gendertext.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump(classifier, f)

    des.append(" ".join(SentenceTokeniser.review_to_wordlist(text, True)))

    # Load  classifer saved  as pickle file
    with open('files/gendertext.pickle','rb') as f:  # Python 3: open(..., 'rb')
           classifier = pickle.load(f)
    # Predict class
    predicted = classifier.predict(des)
    with open('files/mlb.pickle','rb') as f:  # Python 3: open(..., 'rb')
           mlb = pickle.load(f)
    all_labels = mlb.inverse_transform(predicted)
    try :
       return(all_labels[0][0])
    except :
       return "None"
