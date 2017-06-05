"""namesCollection
Authors : Mustafa Atik,Ebin Joshy Nambiaparambil
Version : 1.1
"""
import os


class NamesCollection(object):
    """
    When the first query is received, it reads all the first
    names and their extra information from source file. 
    """

    isInitialized = None

    collectionSourceFile = 'name_gender.csv'
    collection = None

    @classmethod
    def init(cls):
        if not cls.collection:
            cls.collection = cls.loadFromSource()

    @classmethod
    def loadFromSource(cls):
        """Load data from csv"""
        items = {}
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path) + '/files'
        for i in open(dir_path + '/' + cls.collectionSourceFile):

            item = i.strip().split(',')
            firstName = item[0].lower()

            if len(item) == 2:
                item.append('en')

            item = {item[2]:item[1]}

            if firstName in items:
                d = items[firstName].copy()
                d.update(item)
                items[firstName]=d
            else:
                items[firstName] = item

        return items


    @classmethod
    def getGender(cls, firstName, lang='en'):
        """Get the gender of the person"""
        if not cls.isInitialized:
            cls.init()

        firstName = firstName.lower()
        nameInfo = cls.collection.get(firstName, None)
        if not nameInfo:
            return None


        if nameInfo.get('en', None):
            return {'name': firstName, 'gender': nameInfo['en'], 'lang': 'en'}
        else:
            return None
