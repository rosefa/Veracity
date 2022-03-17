import tensorflow_decision_forests as tfdf
from keras.layers.normalization.batch_normalization import BatchNormalization
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
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems
  
def normalize_text(words):
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    words = stem_words(words)
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

def build_bilstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam()
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
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
    #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    
    return model
    
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
def prepare_model_input(train, test,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=300):
    np.random.seed(7)
    text = np.concatenate((train,test), axis=0)
    tokenizer = Tokenizer(num_words=75000)
    tokenizer.fit_on_texts(text)
    sequencesTrain = tokenizer.texts_to_sequences(train)
    sequencesTest = tokenizer.texts_to_sequences(test)
    word_index = tokenizer.word_index
    textTrain = pad_sequences(sequencesTrain, maxlen=300)
    textTest = pad_sequences(sequencesTest, maxlen=300)
    embeddings_dict = {}
    f = open("glove.6B.100d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_dict[word] = coefs
    f.close()
    return (textTrain, textTest, word_index, embeddings_dict)


#data_train, data_test = train_test_split(data, test_size=0.3,shuffle=True)
myData_train, myData_test = train_test_split(dataTest, test_size=0.3,shuffle=True)

trainX = myData_train['article_content']
testX = myData_test['article_content']
trainY = myData_train['labels']
testY = myData_test['labels']
'''trainX = data_train['text']
testX = data_test['text']
trainY = data_train['label']
testY = data_test['label']'''
trainX = [text_prepare(x) for x in trainX]
testX = [text_prepare(x) for x in testX]
myData_train_Glove,myData_test_Glove, word_index, embeddings_dict = prepare_model_input(trainX,testX)
#train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(myData_train, label="labels")
#print(data_test)
#trainX = [text_prepare(x) for x in data_train['text']]
#testX = [text_prepare(x) for x in data_test['text']]

print('pretraitement termine !!!')  

'''embed = "https://tfhub.dev/google/universal-sentence-encoder/4"
embeddings_train = hub.KerasLayer(embed,input_shape=[], dtype=tf.string, trainable=True)
trainX = embeddings_train(trainX)
testX = embeddings_train(testX)
embeddings_train = np.array([np.reshape(embed, (len(embed), 1)) for embed in trainX])
embeddings_test = np.array([np.reshape(embed, (len(embed), 1)) for embed in testX])
print('le model')
model = builModel()
model.fit(embeddings_train,trainY,epochs=10,validation_data=(embeddings_test,testY),batch_size=64,verbose=1)'''
'''train,test = train_test_split(dataTest,test_size=0.3, shuffle=True)'''

optimizer = tf.keras.optimizers.Adam()
input = Input(shape=(300,), dtype='int32')
embedding_matrix = np.random.random((len(word_index)+1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)
model = Conv1D(256, 5,activation='relu')(embedding_layer)
model = MaxPooling1D(2)(model)
model = Conv1D(128, 3,activation='relu')(model)
model = MaxPooling1D(2)(model)
lastLayer = LSTM(64)(model)
outputLayer = Dense(1,activation='sigmoid')(lastLayer)
model = tf.keras.models.Model(inputs=input,outputs=outputLayer)
nn_without_head = tf.keras.models.Model(inputs=model.inputs, outputs=lastLayer)
df_and_nn_model = tfdf.keras.RandomForestModel(preprocessing=nn_without_head)
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    
#model = KerasClassifier(build_bilstm, word_index=word_index, embeddings_dict=embeddings_dict,verbose=0)
#df_and_nn_model = tfdf.keras.RandomForestModel(preprocessing=model)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
history = model.fit(myData_train_Glove, trainY,validation_data=(myData_test_Glove, testY), epochs=50, batch_size=64, verbose=1)
df_and_nn_model.compile(metrics=["accuracy"])
df_and_nn_model.fit(myData_train_Glove,trainY)
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
