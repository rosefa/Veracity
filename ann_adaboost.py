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

def builModel ():
  model = Sequential()
  model.add(layers.Conv1D(128, 5,activation='relu',input_shape=(512, 1)))
  model.add(layers.MaxPooling1D(2))
  model.add(LSTM(32))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
  return model

def build_bilstm(word_index, embeddings_dict, MAX_SEQUENCE_LENGTH=300, EMBEDDING_DIM=100):
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
    
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
def prepare_model_input(X,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=300):
    tokenizer = Tokenizer(num_words=75000)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_Glove = pad_sequences(sequences, maxlen=300)
    word_index = tokenizer.word_index
    
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
    return (X_Glove, word_index, embeddings_dict)

def preprocessing(mitext):
  p = inflect.engine()
  language="en"
  #testp = "(Reuters) - President-elect Donald Trump has chosen former Goldman Sachs partner and Hollywood financier Steven Mnuchin as his nominee for Treasury secretary and billionaire investor Wilbur Ross to head the Commerce Department, the two men told CNBC on Wednesday.    The following is a list of Republican Trumpâ€™s selections for top jobs in his administration: Mnuchin, 53, is a relatively little-known but successful private equity investor, hedge fund manager and Hollywood financier who spent 17 years at Goldman Sachs (GS.N) before leaving in 2002. He assembled an investor group to buy a failed California mortgage lender in 2009, rebranded it as OneWest Bank and built it into Southern Californiaâ€™s largest bank. The bank came under fire for its foreclosure practices as housing advocacy groups accused it of being too quick to foreclose on struggling homeowners. Ross, 78, heads the private equity firm W.L. Ross & Co. His net worth was pegged by Forbes at about $2.9 billion. A staunch supporter of Trump and an economic adviser, Ross has helped shape the Trump campaignâ€™s views on trade policy. He blames the North American Free Trade Agreement with Canada and Mexico, which entered into force in 1994, and the 2001 entry of China into the World Trade Organization for causing massive U.S. factory job losses. Chao, 63, was labor secretary under President George W. Bush for eight years and the first Asian-American woman to hold a Cabinet position. Chao is a director at Ingersoll Rand, News Corp and Vulcan Materials Company. She is married to U.S. Senate Majority Leader Mitch McConnell, a Republican from Kentucky. HEALTH AND HUMAN SERVICES SECRETARY: U.S. REPRESENTATIVE TOM PRICE Price, 62, is an orthopedic surgeon who heads the House of Representativesâ€™ Budget Committee. A representative from Georgia since 2005, Price has criticized Obamacare and has championed a plan of tax credits, expanded health savings accounts and lawsuit reforms to replace it. He is opposed to abortion. U.S. AMBASSADOR TO THE UNITED NATIONS: GOVERNOR NIKKI HALEY Haley, a 44-year-old Republican, has been governor of South Carolina since 2011 and has little experience in foreign policy or the federal government. The daughter of Indian immigrants, Haley led a successful push last year to remove the Confederate battle flag from the grounds of the South Carolina state capitol after the killing of nine black churchgoers in Charleston by a white gunman. DeVos, 58, is a billionaire Republican donor, a former chair of the Michigan Republican Party and an advocate for the privatization of education. As chair of the American Federation for Children, she has pushed at the state level for vouchers that families can use to send their children to private schools and for the expansion of charter schools. [L1N1DO0KC] Sessions, 69, was the first U.S. senator to endorse Trumpâ€™s presidential bid and has been a close ally since. The son of a country-store owner, the senator from Alabama and former federal prosecutor has long taken a tough stance on illegal immigration, opposing any path to citizenship for undocumented immigrants.  NATIONAL SECURITY ADVISER: RETIRED LIEUTENANT GENERAL MICHAEL FLYNN Flynn, 57, was an early supporter of Trump and serves as vice chairman on his transition team. He began his U.S. Army career in 1981 and served deployments in Afghanistan and Iraq. Flynn became head of the Defense Intelligence Agency in 2012 under President Barack Obama, but retired a year earlier than expected, according to media reports, and became a fierce critic of Obamaâ€™s foreign policy. Pompeo, 52, is a third-term congressman from Kansas who serves on the House of Representatives Intelligence Committee, which oversees the CIA, National Security Agency and cyber security. A retired Army officer and Harvard Law School graduate, Pompeo supports the U.S. governmentâ€™s sweeping collection of Americansâ€™ communications data and wants to scrap the nuclear deal with Iran.     "
  seg = pysbd.Segmenter(language="en", clean=False)
  #print(seg.segment(testp))
  nlp = English()
  tokenizer = nlp.tokenizer
  RE = "(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)"
  ps = PorterStemmer()
  mitext2=''
  for sentence in seg.segment(mitext):
    #filtered_sentence = [word for word in [token.text for token in tokenizer(sentence)] if word.lower() not in stopwords.words('english')]
    filtered_sentenceNew = []
    for word in [token.text for token in tokenizer(sentence)] :
      match = re.search(RE, word)
      capital = word.title()
      if match == None or word == capital:
        filtered_sentenceNew.append(word)
    #print(stems)
    tokens_tag = pos_tag(filtered_sentenceNew)
    sentenceTag = []
    for word in tokens_tag : 
      if word[1] in ["NNP","JJ","VB"] and len(word[0])>2 :
        sentenceTag.append(word[0])
    filtered_sentence = [word for word in sentenceTag if word.lower() not in stopwords.words('english')]
    stems = []
    for word in filtered_sentence:
        stem = ps.stem(word)
        stems.append(stem)
    mitext2 = ' '.join([x for x in stems])+ mitext2
  return mitext2

'''**************CROSS VALIDATION********************'''
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cvscores = []
X = dataTest['article_content']
Y = dataTest['labels']
Xpre = preproces.preprocessing(X)
myData_Glove,word_index, embeddings_dict = prepare_model_input(Xpre)
for train, test in kfold.split(myData_Glove,Y):
  model = KerasClassifier(build_bilstm, word_index=word_index, embeddings_dict=embeddings_dict,verbose=0)
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
  history=model.fit(myData_Glove[train], Y[train], validation_data=(myData_Glove[test], Y[test]),epochs=10, batch_size=64, verbose=0)
# evaluate the model
  scores = model.evaluate(myData_Glove[test], Y[test], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
  plot_graphs(history, 'accuracy')
  plot_graphs(history, 'loss')
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
