import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
from SentenceTokeniser import SentenceTokeniser
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers import Input
from keras.datasets import imdb
from keras import backend as K
from keras.models import model_from_json
import h5py
from pathlib import Path
import pickle


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector
def SentimentCheck(text) :
    file1=Path("files/model_new.json")
    # if model_new.jsom doesn't exist
    if(not file1.exists()) :
        print('Indexing word vectors.')
        embeddings_index = {}
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        train=[]
        labels=[]
        with open("Sentiment Analysis Dataset.csv") as f:
                    for row in DictReader(f):
                        label= int(row["Sentiment"])
                        labels.append(label)
                        train.append(row["SentimentText"])

        cleaned_training_data = []
        for i in range( 0, len(train)):
                cleaned_training_data.append(" ".join(SentenceTokeniser.review_to_wordlist(train[i], True)))
        # Save clean_tweets in pickle file
        with open('files/reviews.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump(cleaned_training_data, f)
        with open('files/reviews.pickle','rb') as f:  # Python 3: open(..., 'rb')
               cleaned_training_data = pickle.load(f)
        # print(clean_train_reviews)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(cleaned_training_data)
        sequences = tokenizer.texts_to_sequences(cleaned_training_data)
        word_index = tokenizer.word_index
        test_review=[]
        # text = """A 25-year-old pregnant woman was killed after being hit by a lorry in an accident in Jurong West on Saturday (May 13) morning, Channel NewsAsia understands.
        # The woman, said to be Malaysian national Goh Chieh Ting, was crossing the road at Jurong West Avenue 1 when she was hit by a lorry.
        # The Singapore Police Force said on Sunday that it was alerted to the accident involving a lorry and a pedestrian along Jurong West Avenue 1 towards Corporation Road just before 8am. The victim was sent to Ng Teng Fong General Hospital, where she succumbed to her injuries, police added.
        # Authorities also said that a 46-year-old male lorry driver has been arrested for causing death by a negligent act. Police investigations are ongoing.
        # The Singapore Civil Defence Force, who received an alert at around the same time, said it sent an ambulance to the scene and that the victim was unconscious en route to the hospital.
        # """
        test_review.append(" ".join(SentenceTokeniser.review_to_wordlist(text, True)))
        print("adfad" ,test_review)
        # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        # tokenizer.fit_on_texts(test_review)
        test_sequences = tokenizer.texts_to_sequences(test_review)


        print('Found %s unique tokens.' % len(word_index))
        print(len(sequences))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        test_data= pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print("afsas",test_data)
        labels = to_categorical(labels)
        print("Tasdas",labels[0:3])
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        num_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_NB_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        print('Training model.')
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])

        model.fit(data, labels, batch_size=128, nb_epoch=10, verbose=1)

        model_json = model.to_json()
        with open("files/model_new.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("files/model_new.h5")
        print("Saved model to disk")
    else:
        with open('files/training_data.pickle','rb') as f:  # Python 3: open(..., 'rb')
            cleaned_training_data = pickle.load(f)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(cleaned_training_data)
        test_review=[]
        # text = """A 25-year-old pregnant woman was killed after being hit by a lorry in an accident in Jurong West on Saturday (May 13) morning, Channel NewsAsia understands.
        # The woman, said to be Malaysian national Goh Chieh Ting, was crossing the road at Jurong West Avenue 1 when she was hit by a lorry.
        # The Singapore Police Force said on Sunday that it was alerted to the accident involving a lorry and a pedestrian along Jurong West Avenue 1 towards Corporation Road just before 8am. The victim was sent to Ng Teng Fong General Hospital, where she succumbed to her injuries, police added.
        # Authorities also said that a 46-year-old male lorry driver has been arrested for causing death by a negligent act. Police investigations are ongoing.
        # The Singapore Civil Defence Force, who received an alert at around the same time, said it sent an ambulance to the scene and that the victim was unconscious en route to the hospital.
        # """
        test_review.append(" ".join(SentenceTokeniser.review_to_wordlist(text, True)))
        test_sequences = tokenizer.texts_to_sequences(test_review)
        print("Loaded model from disk")
        test_data= pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    json_file = open('files/model_new.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("files/model_new.h5")
    model.compile(loss='binary_crossentropy',
                  optimizer = 'adam',
                  metrics=["accuracy"])
    # print(model.get_weights())
    print("Loaded model from disk")
    classes = model.predict(test_data)
    print(classes)
    return(classes)
