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

data = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
myData=data.loc[:,'article_content']
labels=data.loc[:,'labels']

#encoded label data
le = preprocessing.LabelEncoder()
le.fit(labels)
myLabel= le.transform(labels)
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
X_train, X_test, Y_train, Y_test = train_test_split(myData, myLabel, test_size = 0.2)
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((Y_train, Y_test), axis=0)

embed = "https://tfhub.dev/google/universal-sentence-encoder/4"

embeddings_train = hub.KerasLayer(embed,input_shape=[], dtype=tf.string, trainable=True)
embeddings_test = hub.KerasLayer(embed,input_shape=[], dtype=tf.string, trainable=True)
train=embeddings_train(X_train)
test=embeddings_test(X_test)
embeddings_train=np.array([np.reshape(embed, (len(embed), 1)) for embed in train])
embeddings_test=np.array([np.reshape(embed, (len(embed), 1)) for embed in test])
acc = []
loss = []
def buldmodel(learn_rate=0.01, momentum=0):
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
    #model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = tf.keras.optimizers.RMSprop(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    return model
# definition du modèle CONV1D et LSTM
seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=buldmodel, epochs=50, batch_size=40, verbose=0)
X = np.concatenate((embeddings_train,embeddings_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)
#batch_size = [5,10, 20, 40, 50,60]
#epochs = [10,20, 50, 60]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model,param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

