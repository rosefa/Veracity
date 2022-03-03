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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
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
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import tensorflow_hub as hub
import statistics

embed = "https://tfhub.dev/google/universal-sentence-encoder/4"
embeddings_train = hub.KerasLayer(embed,input_shape=[], dtype=tf.string, trainable=True)
embeddings_test = hub.KerasLayer(embed,input_shape=[], dtype=tf.string, trainable=True)

#data = pd.concat([dataf1, datav1])


def clean_text(text):
    replaced_text = '\n'.join(s.strip() for s in text.splitlines()[2:] if s != '')  # skip header by [2:]
    replaced_text = replaced_text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', replaced_text)
    #replaced_text = re.sub(r'　', ' ', replaced_text)
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
    return new_words
def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
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
def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas
def normalize_text(words):
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    #words = stem_words(words)
    words = lemmatize_verbs(words)
    words = to_lowercase(words)
    return words
    # Tokenize tweet into words
def tokenize(text):
    return nltk.word_tokenize(text)
#X_train, X_test, Y_train, Y_test = train_test_split(myData, myLabel, test_size = 0.2)
#inputs = np.concatenate((X_train, X_test), axis=0)
#targets = np.concatenate((Y_train, Y_test), axis=0)

model = Sequential()
model.add(Conv1D(256, 5, activation='relu',input_shape=(512, 1)))
model.add(layers.MaxPooling1D(2))
#model.add(BatchNormalization())
model.add(layers.Conv1D(256, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
#model.add(BatchNormalization())
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
#model.add(BatchNormalization())
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(BatchNormalization())
model.add(layers.LSTM(64))
model.add(layers.Dropout(0.2))
#model.add(BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
#formation et évaluation du modèle
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
myDataTest=dataTest.loc[:,'article_content']
labelsTest=dataTest.loc[:,'labels']
dataf1 = pd.read_csv('Pasvrai-1.csv', encoding= 'unicode_escape')
datav1 = pd.read_csv('Vrai-1.csv', encoding= 'unicode_escape')
dataf1 = dataf1.loc[:,'text']
datav1 = datav1.loc[:,'text']
i=0
j=0
myLabel=[]
while i<len(dataf1):
  myLabel.append(0)
  i=i+1
while j<len(datav1):
  myLabel.append(1)
  j=j+1
acc = []
prec = []
rap = []
for p in range(101) :
  dataTrain = np.concatenate((dataf1, datav1), axis=0)
  le = LabelEncoder()
  labelsTrain = le.fit_transform(myLabel)

  train=embeddings_train(dataTrain)
  test=embeddings_test(myDataTest)
  embeddings_train=np.array([np.reshape(embed, (len(embed), 1)) for embed in train])
  embeddings_test=np.array([np.reshape(embed, (len(embed), 1)) for embed in test])
  model.fit(embeddings_train, labelsTrain, epochs=50, batch_size=40, verbose=0)
  results = model.evaluate(embeddings_test, labelsTest, verbose=2)
  print('****************************************************************************')
  acc.append(results[1])
  prec.append(results[2])
  rap.append(results[3])
  #for name, value in zip (model.metrics_names, results) : 
    #print("%s: %.3f" % (name, value))
print(statistics.mean(acc),statistics.mean(prec),statistics.mean(rap))
#batch_size = [5,10, 20, 40, 50,60]
#epochs = [10,20, 50, 60]
#learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#param_grid = dict(learning_rate=learning_rate, momentum=momentum)
#kernel_size = [2, 3,4,5,7]
#param_grid = dict(kernel_size=kernel_size)
#grid = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=-1, cv=5)
#grid_result = grid.fit(X, Y)
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
    #print("%f (%f) with: %r" % (mean, stdev, param))

