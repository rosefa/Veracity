import spacy
import pysbd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy
import wget
#nltk.download('omw-1.4')
import nltk
from numpy import asarray
from numpy import zeros
import inflect
import numpy as np
import pandas as pd
import re
import io
import statistics
import unicodedata
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
#data =dataTest['article_content']
#print (data)
def preprocessing(data):
  ligne =[]
  p = inflect.engine()
  seg = pysbd.Segmenter(language="en", clean=False)
  nlp = English()
  tokenizer = nlp.tokenizer
  RE = "(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)"
  ps = PorterStemmer()
  i=0
  for sentences in data : 
    #mitext = ''
    mitext = []
    
    for sentence in seg.segment(sentences):
      filtered_sentence = []
      for word in [token.text for token in tokenizer(sentence)] :
        #if word.isdigit():
            #word = p.number_to_words(word)
        match = re.search(RE, word)
        capital = word.title()
        if match == None or word == capital:
          filtered_sentence.append(word)
      tokens_tag = pos_tag(filtered_sentence)
      sentenceTag = []
      for word in tokens_tag : 
        if word[1] in ["NNP","JJ","VB"] and len(word[0])>2 :
          sentenceTag.append(word[0])
      filtered_sentenceOtre = [word for word in sentenceTag if word.lower() not in stopwords.words('english')]
      stems = []
      for word in filtered_sentenceOtre:
          stem = ps.stem(word)
          stems.append(stem)
      text = ' '.join([x for x in stems])
      #mitext = mitext+text+' '
      mitext.append(text)
    i=i+1
    #print(i)
    texts = ' '.join([x for x in mitext])
    ligne.append(texts)
    #print (mitext)
  return ligne

def cnn_lstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    optimizer = tf.keras.optimizers.Adam()
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    model = Conv1D(128, 5,activation='relu')(embedding_layer)
    model = MaxPooling1D(2)(model)
    model = LSTM(32)(model)
    model = Dense(1,activation='sigmoid')(model)
    model = keras.Model(inputs=input,outputs=model)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model   
  
def cnn_bilstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    optimizer = tf.keras.optimizers.Adam()
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    model = Conv1D(128, 5,activation='relu')(embedding_layer)
    model = MaxPooling1D(2)(model)
    model = Bidirectional(LSTM(32))(model)
    model = Dense(1,activation='sigmoid')(model)
    model = keras.Model(inputs=input,outputs=model)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model
  
 def cnn_mtm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    optimizer = tf.keras.optimizers.Adam()
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    model = Conv1D(128, 5,activation='relu')(embedding_layer)
    model = MaxPooling1D(2)(model)
    model = Dense(128,activation='relu')(model)
    model = Dense(32,activation='relu')(model)
    model = Dense(1,activation='sigmoid')(model)
    model = keras.Model(inputs=input,outputs=model)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model
 
def dcnn_lstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    optimizer = tf.keras.optimizers.Adam()
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    model = Conv1D(128, 5,activation='relu')(embedding_layer)
    model = Conv1D(100, 5,activation='relu')(embedding_layer)
    model = MaxPooling1D(2)(model)
    model = Lstm(32)(model)
    model = keras.Model(inputs=input,outputs=model)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model
