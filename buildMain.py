Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@rosefa 
rosefa
/
Veracity
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Veracity/ann_adaboost.py /
@rosefa
rosefa Update ann_adaboost.py
Latest commit 0c8c055 21 hours ago
 History
 1 contributor
225 lines (214 sloc)  11.8 KB
   
import preprocessing as preproces
import spacy
import pysbd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import tensorflow_decision_forests as tfdf
from keras.layers.normalization.batch_normalization import BatchNormalization
import wget
#nltk.download('omw-1.4')
import nltk
from numpy import asarray
from numpy import zeros
from sklearn.preprocessing import LabelEncoder
import inflect
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Layer
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
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#nltk.download('averaged_perceptron_tagger')


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
data = pd.concat([dataf1,dataf2,dataf3,datav1,datav2,datav3,datav4], axis=0)
#print(list(data.columns))
dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


'''**************CROSS VALIDATION********************'''
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
X = dataTest['article_content']
Y = dataTest['labels']
Xpre = preproces.preprocessing(X)
myData_Glove,word_index, embeddings_dict = prepare_model_input(Xpre)
for train, test in kfold.split(myData_Glove,Y):
  model = preproces.cnn_bilstm(word_index=word_index, embeddings_dict=embeddings_dict)
  history=model.fit(myData_Glove[train], Y[train], validation_data=(myData_Glove[test], Y[test]),epochs=10, batch_size=64, verbose=0)
  scores = model.evaluate(myData_Glove[test], Y[test], verbose=1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
  print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
  cvscores.append(scores[1] * 100)
  plot_graphs(history, 'accuracy')
  plot_graphs(history, 'loss')
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
