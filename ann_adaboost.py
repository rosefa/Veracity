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

#embed = "https://tfhub.dev/google/universal-sentence-encoder/4"

dataf1 = pd.read_csv('Pasvrai-1.csv', encoding= 'unicode_escape')
dataf2 = pd.read_csv('Pasvrai-2.csv', encoding= 'unicode_escape')
dataf3 = pd.read_csv('Pasvrai-3.csv', encoding= 'unicode_escape')
datav1 = pd.read_csv('True-1.csv', encoding= 'unicode_escape')
datav2 = pd.read_csv('True-2.csv', encoding= 'unicode_escape')
datav3 = pd.read_csv('True-3.csv', encoding= 'unicode_escape')
datav4 = pd.read_csv('True-4.csv', encoding= 'unicode_escape')
neg =[]
pos =[]
i=0
while i<len(dataf1):
  neg.append(0)
  i=i+1
i=0
while i<len(dataf2):
  neg.append(0)
  i=i+1
i=0
while i<len(dataf3):
  neg.append(0)
  i=i+1
i=0
while i<len(datav1):
  pos.append(1)
  i=i+1
i=0
while i<len(datav2):
  pos.append(1)
  i=i+1
i=0
while i<len(datav3):
  pos.append(1)
  i=i+1
i=0
while i<len(datav4):
  pos.append(1)
  i=i+1

#dataf1['label']=neg
#datav1['label']=neg
#data = pd.concat([dataf1['text'], datav1['text']])

#print (data.shape)

print (dataf1.shape)
print (dataf2.shape)
print (dataf3.shape)
print (datav1.shape)
print (datav2.shape)
print (datav3.shape)
print (datav4.shape)
#print(labels)
