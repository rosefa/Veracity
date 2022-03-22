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

dataTest = pd.read_csv('FAKESDataset.csv', encoding= 'unicode_escape')
def preprocessing(data):
  p = inflect.engine()
  language="en"
  #testSentences = "Wed 05 Apr 2017 Syria attack symptoms consistent with nerve agent use WHO. Victims of a suspected chemical attack in Syria appeared to show symptoms consistent with reaction to a nerve agent the World Health Organization said on Wednesday. ""Some cases appear to show additional signs consistent with exposure to organophosphorus chemicals a category of chemicals that includes nerve agents"" WHO said in a statement putting the death toll at at least 70. The United States has said the deaths were caused by sarin nerve gas dropped by Syrian aircraft. Russia has said it believes poison gas had leaked from a rebel chemical weapons depot struck by Syrian bombs. Sarin is an organophosporus compound and a nerve agent. Chlorine and mustard gas which are also believed to have been used in the past in Syria are not. A Russian Defence Ministry spokesman did not say what agent was used in the attack but said the rebels had used the same chemical weapons in Aleppo last year. The WHO said it was likely that some kind of chemical was used in the attack because sufferers had no apparent external injuries and died from a rapid onset of similar symptoms including acute respiratory distress. It said its experts in Turkey were giving guidance to overwhelmed health workers in Idlib on the diagnosis and treatment of patients and medicines such as Atropine an antidote for some types of chemical exposure and steroids for symptomatic treatment had been sent. A U.N. Commission of Inquiry into human rights in Syria has previously said forces loyal to Syrian President Bashar al-Assad have used lethal chlorine gas on multiple occasions. Hundreds of civilians died in a sarin gas attack in Ghouta on the outskirts of Damascus in August 2013. Assads government has always denied responsibility for that attack. Syria agreed to destroy its chemical weapons in 2013 under a deal brokered by Moscow and Washington. But Russia a Syrian ally and China have repeatedly vetoed any United Nations move to sanction Assad or refer the situation in Syria to the International Criminal Court. ""These types of weapons are banned by international law because they represent an intolerable barbarism"" Peter Salama Executive Director of the WHO Health Emergencies Programme said in the WHO statement. - REUTERS"
  seg = pysbd.Segmenter(language="en", clean=False)
  nlp = English()
  tokenizer = nlp.tokenizer
  RE = "(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)"
  ps = PorterStemmer()
  ligne=[]
  for row in data : 
    for sentences in row : 
      mitext2=''
      for sentence in seg.segment(sentences):
        filtered_sentence = []
        for word in [token.text for token in tokenizer(sentence)] :
          match = re.search(RE, word)
          capital = word.title()
          if match == None or word == capital:
            filtered_sentence.append(word)
        tokens_tag = pos_tag(filtered_sentence)
        sentenceTag = []
        for word in tokens_tag : 
          if word[1] in ["NNP","NNPS","NN","JJ","VB"] and len(word[0])>2 :
            sentenceTag.append(word[0])
        filtered_sentenceOtre = [word for word in sentenceTag if word.lower() not in stopwords.words('english')]
        stems = []
        for word in filtered_sentenceOtre:
            stem = ps.stem(word)
            stems.append(stem)
        text = ' '.join([x for x in stems])
        mitext2 = mitext2+text
      print(mitext2)
      ligne.append(mitext2)
       
  #return ligne
print(preprocessing(dataTest))
