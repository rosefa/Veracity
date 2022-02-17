# -*- coding: utf-8 -*-
"""VDC-BILSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N-AZ53MiABekya1Zn2fND54gRu2mqL9W
"""

#importing required libraries
import wget
import keras
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import nltk
#nltk.download('omw-1.4')
import inflect
import contractions
from bs4 import BeautifulSoup
#import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional,Conv1D,MaxPooling1D,Flatten,Activation,GlobalMaxPooling1D,LeakyReLU,concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Input, Layer
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
#import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import logging
import io
from zipfile import ZipFile
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
#from google.colab import files
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#nltk.download('punkt')
#from nltk.corpus import punkt
from nltk.corpus import wordnet
from nltk.corpus import stopwords

url= 'http://nlp.stanford.edu/data/glove.6B.zip'
filename = wget.download(url)
with ZipFile(filename, 'r') as f:
    f.extractall()
#unzip filename

#***************NETTOYAGE DES DONNEES***********************
# First function is used to denoise text
def clean_text(text):
    replaced_text = '\n'.join(s.strip() for s in text.splitlines()[2:] if s != '')  # skip header by [2:]
    replaced_text = replaced_text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', replaced_text)
    replaced_text = re.sub(r'　', ' ', replaced_text)
    return replaced_text

def clean_url(html_text):
    clean_text = re.sub(r'http\S+', '', html_text)
    return clean_text
def clean_html_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    new_words = ' '.join([x for x in new_words])
    return new_words
def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    new_words = ' '.join([x for x in new_words])
    return new_words
#***************NORMALISATION***********************
def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text
'''def lemmatize_term(term, pos=None):
    if pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(term, pos=pos)'''
def lemmatize_text(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        pos=None
        if pos is None:
            synsets = wordnet.synsets(word)
            if not synsets:
                lemmas.append(word)
            else :
                pos = synsets[0].pos()
                if pos == wordnet.ADJ_SAT:
                    pos = wordnet.ADJ
                lemma = lemmatizer.lemmatize(word, pos=pos)
                lemmas.append(lemma)
    lemmas = ' '.join([x for x in lemmas])
    return lemmas  
        
def normalize_number(text):
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text
#**********************PRETRAITEMENT**************************
def text_prepare(text):
    text_traite = clean_text(text)
    text_traite = clean_url(text_traite)
    text_traite = clean_html_tags(text_traite)
    text_traite = remove_stopwords(text_traite)
    text_traite = remove_punctuation(text_traite)
    text_traite = normalize_unicode(text_traite,form='NFKC')
    text_traite = normalize_number(text_traite)
    text_traite = lemmatize_text(text_traite)
    #text_traite = clean_url(text)
    #text_traite = clean_url(text)
    #mitext = ' '.join([x for x in lemmatize_text(text_traite)])
    return text_traite
'''def clean_html_and_js_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text'''

'''def denoise_text(text):
    text = BeautifulSoup(text)
    text = str (text.get_text().encode())
    return text'''
# Check the function 

# Text normalization includes many steps.
# Each function below serves a step.
'''def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words'''
'''def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words'''
'''def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words'''
'''def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems
def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas'''

'''def normalize_text(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    return words'''
    # Tokenize tweet into words
'''def tokenize(text):
    return nltk.word_tokenize(text)'''
# check the function



'''myData = [text_prepare(x) for x in myData]
le = LabelEncoder()
mylabels = le.fit_transform(mylabels)'''

#************************ VECTORISATION DU TEXT**************************

def prepare_model_input(X_train, X_test,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=300):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    #text = pd.concat((X_train, X_test))
    #text = np.array(text)
    tokenizer = Tokenizer(num_words=75000)
    tokenizer.fit_on_texts(text)
    # pickle.dump(tokenizer, open('text_tokenizer.pkl', 'wb'))
    # Uncomment above line to save the tokenizer as .pkl file 
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=300)
    #print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    #print(text.shape)
    X_train_Glove = text[0:len(X_train), ]
    X_test_Glove = text[len(X_train):, ]
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
    #print('Total %s word vectors.' % len(embeddings_dict))
    return (X_train_Glove, X_test_Glove, word_index, embeddings_dict)

#********************* CONSTRUCTION DES MODELS *****************************
def cnn_bilstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100,merge_mode="sum"):
    # Initialize a sequebtial model
    '''accuracy = tf.keras.metrics.Accuracy(name='accuracy')
    precision = tf.keras.metrics.Precision(name='precision')
    rappel = tf.keras.metrics.Recall(name='recall')
    model = Sequential()
    # Make the embedding matrix using the embedding_dict
    embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
            
    # Add embedding layer
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    # Add hidden layers
    conv=Conv1D(128, 5,activation='relu')(embedding_layer)
    pool =MaxPooling1D(2)(conv)
    conv=Conv1D(128, 5,activation='relu')(pool)
    pool =MaxPooling1D(2)(conv)
    conv=Conv1D(128,5 ,activation='relu')(pool)
    pool =MaxPooling1D(2)(conv)
    flat = Flatten()(pool)
    dense = Dense(128)(flat)
    dense = Dropout(0.5)(dense)
    #rnn_layer = LSTM(128, batch_input_shape = (10, 300,))(embedding_layer, initial_state = [model1, model1])
    bi = Bidirectional(LSTM(128,recurrent_dropout=0.2))(embedding_layer)
    densebi = concatenate([dense,bi])
    attention_prob = Dense(128,activation = 'softmax')(densebi)
    attention_mul = concatenate([densebi, attention_prob])
    attention_mul = Dropout(0.2)(attention_mul)
    #model1 = Bidirectional(LSTM(32,recurrent_dropout=0.2),merge_mode=merge_mode)(model1)
    preds = Dense (1, activation = 'sigmoid')(attention_mul)
    reLU = LeakyReLU (alpha = 0.1)(preds)
    model = keras.Model(inputs=input,outputs=reLU)
    #plot_model(model, "VDC-BILSTM.png", show_shapes=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model

def cnn_lstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100,merge_mode="sum"):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    # Add hidden layers 
    model1=Conv1D(128, 5,activation='relu')(embedding_layer)
    model1 =MaxPooling1D(2)(model1)
    model1 = Dropout(0.2)(model1)
    model1 = LSTM(32)(model1)
    model1 = Dense(1,activation='sigmoid')(model1)
    model = keras.Model(inputs=input,outputs=model1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model

def dlstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100,merge_mode="sum"):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    # Add hidden layers 
    model1 = LSTM(128,return_sequences=True)(embedding_layer)
    model1 = Dropout(0.2)(model1)
    model1 = LSTM(128)(model1)
    model1 = Dense(256,activation='relu')(model1)
    model1 = Dropout(0.2)(model1)
    model1 = Dense(1,activation='sigmoid')(model1)
    model = keras.Model(inputs=input,outputs=model1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model
def dann(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100,merge_mode="sum"):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)
    # Add hidden layers 
    model1 = Dropout(0.2)(embedding_layer)
    model1 = GlobalMaxPooling1D()(model1)
    model1 = Dropout(0.2)(model1)
    model1 = Dense(256,activation='relu')(embedding_layer)
    model1 = Dropout(0.2)(model1)
    model1 = Dense(128,activation='relu')(model1)
    model1 = Dropout(0.2)(model1)
    model1 = Dense(64,activation='relu')(model1)
    model1 = Dropout(0.2)(model1)
    model1 = Dense(32,activation='relu')(model1)
    model1 = Dropout(0.2)(model1)
    model1 = Dense(1,activation='sigmoid')(model1)
    model = keras.Model(inputs=input,outputs=model1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return model

def vdc_lstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)       

   
    
    model1=Conv1D(32, 3,activation='relu')(embedding_layer)
    model1= Conv1D(32, 3,activation='relu')(model1)
    model1 = Activation('relu')(model1)
    model1 =MaxPooling1D(2)(model1)
    model1 = Dropout(0.2)(model1)
    model1=Conv1D(64, 5,activation='relu')(embedding_layer)
    model1= Conv1D(64, 5,activation='relu')(model1)
    model1 = Activation('relu')(model1)
    model1 =MaxPooling1D(2)(model1)
    model1 = Dropout(0.2)(model1)
    model1= Conv1D(128,5,activation='relu')(model1)
    model1= Conv1D(128,5,activation='relu')(model1)
    model1 = Activation('relu')(model1)
    model1 =MaxPooling1D(2)(model1)
    model1 = Dropout(0.2)(model1)
    model1= Conv1D(256,7,activation='relu')(model1)
    model1= Conv1D(256,7,activation='relu')(model1)
    model1 = Activation('relu')(model1)
    model1 =MaxPooling1D(2)(model1)
    model1 = Dropout(0.2)(model1)
    #model1= GlobalMaxPooling1D()(model1)
    model1 = LSTM(64,return_sequences=True)(model1)
    model1 = Dropout(0.2)(model1)
    model1 = LSTM(64)(model1)
    model1 = Dense(1,activation='sigmoid')(model1)
    model = keras.Model(inputs=input,outputs=model1)
    #plot_model(model, "VDC-BILSTM.png", show_shapes=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    
    return model

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = (2*(precision*recall))/(precision+recall)
    return {
        "mcc": mcc,
        "true positive": tp,
        "true negative": tn,
        "false positive": fp,
        "false negative": fn,
        "precision" : precision,
        "recall" : recall,
        "F1" : f1,
        "accuracy": (tp+tn)/(tp+tn+fp+fn)
    }
def compute_metrics(labels, preds):
    #assert len(preds) == len(labels)
    return get_eval_report(labels, preds)
def plot_graphs(history1,history2,history3,history5, string):
  plt.plot(history1.history[string],'r-',history2.history[string],'b-',history3.history[string],'o-',history5.history[string],'k-')
  #plt.plot(history.history['val_'+string], 'r--',history2.history['val_'+string],'r-')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend(['CNN-BILSTM','CNN-LSTM','VDC-LSTM','DANN','DLSTM'])
  plt.show()

def plot_graphs2(history1,history2,history3,history5, string):
  #plt.plot(history.history[string],'b--',history2.history[string],'b-',history3.history[string],'b-',history5.history[string],'b-')
  plt.plot(history1.history[string], 'r-',history2.history[string],'b-',history3.history[string],'o-',history5.history[string],'k-')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  #plt.legend([string, 'val_'+string])
  plt.legend(['CNN_BILSTM','CNN_LSTM','VDC_LSTM','DANN','DLSTM'])
  plt.show()
#*********************** DEBUT DU PROCESSING***********************
logging.basicConfig(level=logging.INFO)
labels=[]
data = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
myDatatest=data.loc[:,'article_content']
labelstest=data.loc[:,'labels']
dataf1 = pd.read_csv('Pasvrai-1.csv', encoding= 'unicode_escape')
datav1 = pd.read_csv('Vrai-1.csv', encoding= 'unicode_escape')

i=0
j=0
while i<len(dataf1):
  labels.append(0)
  i=i+1
while j<len(datav1):
  labels.append(1)
  j=j+1
data = pd.concat([dataf1['text'], datav1['text']])
'''print(data.head())
print(len(data))
print(len(labels))'''
#myData=data
myData=data
labels=labels
'''titre= data.loc[:,'article_title']
myData=data.loc[:,"text"]
mylabels=data.loc[:,'labels']'''
myData = [text_prepare(x) for x in myData]
print('pretraitement termine !!!')
le = LabelEncoder()
mylabels = le.fit_transform(labels)
X = myData
y = mylabels
kf = KFold(n_splits=10)
losses = []
exactitudeTab = []
precisionTab = []
rappelTab = []
histoire=[]
seed = 7
np.random.seed(seed)
#myDatatest=data.loc[:,'article_content']
#labelstest=data.loc[:,'labels']
x_train,x_test,y_train,y_test = train_test_split(myData,mylabels, test_size=0.2)
#myData_train_Glove,myData_test_Glove, word_index, embeddings_dict = prepare_model_input(myData,myDatatest)
myData_train_Glove,myData_test_Glove, word_index, embeddings_dict = prepare_model_input(x_train,x_test)
textData = np.concatenate((myData_train_Glove, myData_test_Glove), axis=0)
textLabel = np.concatenate((y_train, y_test), axis=0)
print("debut des k-fold")
#text = myData_train_Glove
#mylabels = mylabels
#myDatatest = myData_test_Glove
#mylabels = np.concatenate((y_train, y_test), axis=0)
'''#model = build_bilstm(word_index, embeddings_dict, 1)
#model = KerasClassifier(build_fn=build_bilstm(word_index, embeddings_dict, 1), verbose=0)
model = KerasClassifier(build_fn=build_bilstm, word_index=word_index, embeddings_dict=embeddings_dict,batch_size=64,epochs=10,verbose=0)
# define the grid search parameters
#batch_size = [60, 80, 100,150]
#epochs = [10,50,60,100]
merge_mode=['sum', 'mul', 'concat', 'ave']
#optimizer = ['adam']
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(merge_mode=merge_mode)
#param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(text, mylabels)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))'''

kf = KFold(n_splits=5)
for train, test in kf.split(textData,textLabel) :
  model = cnn_bilstm(word_index, embeddings_dict)
  history1 = model.fit(textData[train], textLabel[train],validation_split=0.2, epochs=10, batch_size=64, verbose=0)
  results1 = model.evaluate(textData[test],textLabel[test],verbose=0)
  model = cnn_lstm(word_index, embeddings_dict)
  history2 = model.fit(textData[train], textLabel[train],validation_split=0.2, epochs=10, batch_size=64,verbose=0)
  results2 = model.evaluate(textData[test],textLabel[test],verbose=0)
  model = vdc_lstm(word_index, embeddings_dict)
  history3 = model.fit(textData[train], textLabel[train],validation_split=0.2, epochs=10, batch_size=64, verbose=0)
  results3 = model.evaluate(textData[test],textLabel[test],verbose=0)
  #model = dann(word_index, embeddings_dict)
  #history4 = model.fit(text[train], mylabels[train],validation_data=(myDatatest,labelstest), epochs=10, batch_size=64, verbose=0)
  #results4 = model.evaluate(text[test],mylabels[test],verbose=0)
  model = dlstm(word_index, embeddings_dict)
  history5 = model.fit(textData[train], textLabel[train],validation_split=0.2, epochs=10, batch_size=64, verbose=0)
  results5 = model.evaluate(textData[test],textLabel[test],verbose=0)
  model = dann(word_index, embeddings_dict)
  #history6 = model.fit(textData[train], textLabel[train],validation_split=0.2, epochs=10, batch_size=64, verbose=0)
  #results6 = model.evaluate(textData[test],textLabel[test],verbose=0)
  plot_graphs(history1, history2,history3,history5,'accuracy')
  plot_graphs2(history1, history2,history3,history5,'val_accuracy')
  plot_graphs(history1,history2,history3,history5, 'loss')
  #exactitudeTab.append(results[1])
  #precisionTab.append(results[2])
  #rappelTab.append(results[3])
  print('cnn_bilstm')
  print(results1[1])
  print(results1[2])
  print(results1[3])
  print('cnn_lstm')
  print(results2[1])
  print(results2[2])
  print(results2[3])
  print('vdc_lstm')
  print(results3[1])
  print(results3[2])
  print(results3[3])
  #print('dann')
  #print(results4[1])
  #print(results4[2])
  #print(results4[3])
  print('dlstm')
  print(results5[1])
  print(results5[2])
  print(results5[3])
  print('dann')
  #print(results6[1])
  #print(results6[2])
  #print(results6[3])
  print('******************************************************') 
