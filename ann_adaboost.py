from keras.layers.normalization.batch_normalization import BatchNormalization
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip -q glove.6B.zip
#!ls
#!pip install scikeras
#!pip install --upgrade tensorflow_hub
import wget
#nltk.download('omw-1.4')
from numpy import asarray
from numpy import zeros
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM, GRU
from keras.layers import Bidirectional
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
#import tensorflow_decision_forests as tfdf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
#from keras.wrappers.scikit_learn import KerasRegressor
#from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf 
import io
#import chardet
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import tensorflow_hub as hub
import statistics
import unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

dataf1 = pd.read_csv('Pasvrai-1.csv', encoding= 'unicode_escape')
dataf2 = pd.read_csv('Pasvrai-2.csv', encoding= 'unicode_escape')
dataf3 = pd.read_csv('Pasvrai-3.csv', encoding= 'unicode_escape')
datav1 = pd.read_csv('True-1.csv', encoding= 'unicode_escape')
datav2 = pd.read_csv('True-2.csv', encoding= 'unicode_escape')
datav3 = pd.read_csv('True-3.csv', encoding= 'unicode_escape')
datav4 = pd.read_csv('True-4.csv', encoding= 'unicode_escape')
neg =[]
i=0
while i<len(dataf1):
  neg.append(0)
  i=i+1
dataf1['label']=neg
i=0
neg =[]
while i<len(dataf2):
  neg.append(0)
  i=i+1
dataf2['label']=neg
i=0
neg =[]
while i<len(dataf3):
  neg.append(0)
  i=i+1
dataf3['label']=neg
pos =[]
i=0
while i<len(datav1):
  pos.append(1)
  i=i+1
datav1['label']=pos
pos =[]
i=0
while i<len(datav2):
  pos.append(1)
  i=i+1
datav2['label']=pos
pos =[]
i=0
while i<len(datav3):
  pos.append(1)
  i=i+1
datav3['label']=pos
pos =[]
i=0
while i<len(datav4):
  pos.append(1)
  i=i+1
datav4['label']=pos
data = pd.concat([dataf1,dataf2,dataf3,datav1,datav2,datav3,datav4])
'''************** preprocessing****************'''
def clean_text(text):
    
    replaced_text = re.sub(r'[【】]', ' ', text)
    replaced_text = re.sub(r'[（）()]', ' ', text)
    replaced_text = re.sub(r'[［］\[\]]', ' ', text)
    replaced_text = re.sub(r'[@＠]\w+', '',text)
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', text)
    return replaced_text 

# Text normalization includes many steps.
# Each function below serves a step.
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def normalize_text(words):
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words
    # Tokenize tweet into words
def text_prepare(text):
    mitext = clean_text(text)
    mitext = ' '.join([x for x in normalize_text(mitext)])
    return mitext
  
def builModel ():
  model = Sequential()
  model.add(layers.Conv1D(128, 5,activation='relu',input_shape=(512, 1)))
  model.add(layers.MaxPooling1D(2))
  model.add(LSTM(32))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
  return model

data = [text_prepare(x) for x in data]
print('pretraitement termine !!!')  
data_train, data_test = train_test_split(data, test_size=0.3,shuffle=True)
trainX = [text_prepare(x) for x in data_train.text]
testX = [text_prepare(x) for x in data_test.text]
trainY = [text_prepare(x) for x in data_train.label]
testY = [text_prepare(x) for x in data_test.label]
embed = "https://tfhub.dev/google/universal-sentence-encoder/4"
embeddings_train = hub.KerasLayer(embed,input_shape=[], dtype=tf.string, trainable=True)
trainX = embeddings_train(trainX)
testX = embeddings_train(testX)
embeddings_train=np.array([np.reshape(embed, (len(embed), 1)) for embed in trainX])
embeddings_test=np.array([np.reshape(embed, (len(embed), 1)) for embed in testX])
model = builModel()
model.fit(embeddings_train,trainY,epochs=10,validation_data=(embeddings_test,testY),batch_size=64,verbose=1)
