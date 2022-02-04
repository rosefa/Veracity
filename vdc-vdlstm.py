import wget
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import nltk
nltk.download('omw-1.4')
import inflect
import contractions
from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional,Conv1D,MaxPooling1D,Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
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
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
url= 'http://nlp.stanford.edu/data/glove.6B.zip'
filename = wget.download(url)
with ZipFile(filename, 'r') as f:
    f.extractall()
#unzip filename
logging.basicConfig(level=logging.INFO)
#uploaded = files.upload()
labels=[]
data = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
myData=data.loc[:,'article_content']
labels=data.loc[:,'labels']
#data1 = pd.read_csv('Fake.csv',usecols= ['text'] )
#print('data1 ok')
#data2 = pd.read_csv('True.csv',usecols= ['text'] )
#print('data2 ok')

# First function is used to denoise text
def denoise_text(text):
    # Strip html if any. For ex. removing <html>, <p> tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Replace contractions in the text. For ex. didn't -> did not
    text = contractions.fix(text)
    return text
# Check the function 

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
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    return words
    # Tokenize tweet into words
def tokenize(text):
    return nltk.word_tokenize(text)
# check the function

def text_prepare(text):
    mitext = denoise_text(text)
    mitext = ' '.join([x for x in normalize_text(tokenize(mitext))])
    return mitext

def prepare_model_input(X_train, X_test,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=300):
#def prepare_model_input(X_train,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=300):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    tokenizer = Tokenizer(num_words=75000)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=300)
    #print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    #np.random.shuffle(indices)
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
    return (X_train_Glove, X_test_Glove, word_index, embeddings_dict)
    #return (text, word_index, embeddings_dict)

def build_bilstm(word_index, embeddings_dict, optimizer='adam', MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100, dropout=0.5, hidden_layer = 3, lstm_node = 32):
    input = Input(shape=(300,), dtype='int32')
    embedding_matrix = np.random.random((len(word_index)+1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=300,trainable=True)(input)       

    model1=Conv1D(128, 5, activation="relu")(embedding_layer)
    #model1= MaxPooling1D(2)(model1)
    model1= Conv1D(128, 5, activation="relu")(model1)
    model1 = BatchNormalization()(model1)
    #model1= MaxPooling1D(2)(model1)
    model1= Conv1D(256,5,activation='relu')(model1)
    model1 = BatchNormalization()(model1)
    model1= GlobalMaxPooling1D()(model1)
    #model1= Dropout(0.5)(model1)
    model1= Dense(256,activation='relu')(model1)
    #model1= Dropout(0.5)(model1)

    model2 = Bidirectional(LSTM(128))(embedding_layer)
    model2 = Dropout(0.2)(model2)
    #model2 = Flatten()(model2)
    model2= Dense(256,activation='relu')(model2)
  
    
    #model3 = layers.maximum([model1,model2])
    model3 = layers.concatenate([model1,model2])
    model3 = Dense(512, activation='relu')(model3)
    #model3 = Dropout(0.5)(model3)
    model3 = Dense(1, activation='sigmoid')(model3)

    model = keras.Model(inputs=input,outputs=model3)
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
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

kf = KFold(n_splits=5)
losses = []
exactitudeTab = []
precisionTab = []
rappelTab = []
histoire=[]

print("Preparing model input ...")
print("Done!")
print("Building Model!")


myData = [text_prepare(x) for x in myData]
#myData2 = [text_prepare(x) for x in myData2]
le = LabelEncoder()
mylabels = le.fit_transform(labels)
#mylabels2 = le.fit_transform(labels2)
lr_init=0.01
epochs=10
decay=lr_init/epochs
def lr_decay(epoch,lr):
  drop_rate=0.5
  epochs_drop=10.0
  #return lr_init*1/(1+decay*epoch)
  return lr_init*math.pow(drop_rate, math.floor(100/epochs_drop))
print("traitement éffectué. debut du machine learning...")
seed = 7
np.random.seed(seed)
x_train,x_test,y_train,y_test = train_test_split(myData,mylabels, test_size=0.2)
myData_train_Glove,myData_test_Glove, word_index, embeddings_dict = prepare_model_input(x_train,x_test)
text = np.concatenate((myData_train_Glove, myData_test_Glove), axis=0)
#model = build_bilstm(word_index, embeddings_dict, 1)
#model = KerasClassifier(build_fn=build_bilstm(word_index, embeddings_dict, 1), verbose=0)
model = KerasClassifier(build_bilstm, word_index=word_index, embeddings_dict=embeddings_dict, epochs=50, batch_size=10,verbose=0)
# define the grid search parameters
#batch_size = [10, 20, 40, 60, 80, 100,150]
#epochs = [10,50]
optimizer = ['sgd', 'rmsprop','adam']
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
#param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(text, mylabels)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''history = model.fit(myData_train_Glove, y_train,validation_split=0.1, epochs=60, batch_size=150, verbose=2)

resultsTrain = model.evaluate(myData_train_Glove, y_train,verbose=0)
results = model.evaluate(myData_test_Glove, y_test,verbose=0)
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
print(results[1],resultsTrain[1])
print(results[2],resultsTrain[2])
print(results[3],resultsTrain[3])'''
'''for train, test in kf.split(myData_train_Glove,mylabels) :
  model = build_bilstm(word_index, embeddings_dict, 1)
  history = model.fit(myData_train_Glove[train], mylabels[train],validation_data=(myData_train_Glove[test],mylabels[test]), epochs=10, batch_size=64, verbose=1)
  results = model.evaluate(myData_train_Glove[test], mylabels[test],verbose=0)
  plot_graphs(history, 'accuracy')
  plot_graphs(history, 'loss')
  exactitudeTab.append(results[1])
  precisionTab.append(results[2])
  rappelTab.append(results[3])
  print(results[1])
  print(results[2])
  print(results[3])
  print(i+1) 
  print('****************************')
meanAcc = np.mean(exactitudeTab)
meanExa = np.mean(precisionTab)
meanRap = np.mean(rappelTab)
print(meanAcc)
print(np.std(exactitudeTab))
print(meanExa)
print(np.std(precisionTab))
print(meanRap)
print(np.std(rappelTab))
print(2*(meanExa*meanRap)/(meanExa+meanRap))'''



'''print("\n Evaluating Model ... \n")
#predicted = model.predict_classes(X_test_Glove)
predicted=model.predict(X_test_Glove) 
#print(metrics.classification_report(y_test, predicted))
print("\n")
logger = logging.getLogger("logger")
result = compute_metrics(y_test, predicted)
for key in (result.keys()):
    logger.info("  %s = %s", key, str(result[key]))'''

#To save the tokenizer follow instructions in prepare_model_input function i.e. uncomment this line #pickle.dump(tokenizer, open('text_tokenizer.pkl', 'wb')) in that function
# To save the model run this line
#pickle.dump(model, open('model.pkl', 'wb'))
# you are ready for deployment!