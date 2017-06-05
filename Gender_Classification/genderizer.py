#!/usr/bin/python3
"""genderizer
For finding the gender on the basis of firstname , description and text
"""

import gender as g
from namesCollection import NamesCollection as n

male = 'M'
female = 'F'

def detect(firstName=None, text=[], description=None):
""" Function to find the geder from firstName ,text or description"""


    if firstName:
        nameGender = n.getGender(firstName)
        """ If the first name surely is used for only one gender,
            we can accept this gender.Otherwise run SVM classification over
            description and text
        """

        if nameGender and (nameGender['gender']==male or nameGender['gender'] == female):
            if nameGender['gender'] == male:
                return 'male'
            elif nameGender['gender'] == female:
                return 'female'
        elif description!=None :
            #check 'description' for gender
            ret=g.checkDescription(description)
            if ret==None and len(text)>0:
                #check tweets for gender
                ret=g.checkTweet(text)
            if(ret!=None) :
               return ret
            else :
               return 'brand'
        else:
           return 'brand'
    else :
         if description!=None :
             #check 'description' for gender
             ret=g.checkDescription(description)
             if ret==None and len(text)>0:
                 #check tweets for gender
                 ret=g.checkTweet(text)
             if(ret!=None) :
                 return ret
             else :
                return 'brand'
         else:
            return 'brand'
