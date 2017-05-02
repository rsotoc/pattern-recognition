# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:00:34 2017

@author: rsotoc
"""

import numpy as np
import pandas as pd
import time

import re
import nltk
from nltk.corpus import stopwords 
from nltk.util import ngrams
from sklearn.model_selection import train_test_split

from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB

def document_features_ngrams(document, global_features): 
    features = [0] * len(global_features)
    nd = len(document)
    for elem in document:
        if elem in global_features:
            tf = document.freq(elem) / nd
            index, idf = global_features[elem]
            if ( idf > 0 ):
                features[index] = tf * idf
    return features


# ------------------------------------------------------------------
movies_reviews = pd.read_csv("Movies Reviews/labeledTrainData.tsv", sep='\t')

# Limpiar los documentos. Conservar sólo plabras (alfabéticas) y pasar a minúsculas
movies_reviews.review = list(map(lambda row: re.sub("[^a-zA-Z]", " ", 
                                BeautifulSoup(row, "lxml").get_text().lower()), 
                                 movies_reviews.review))

# Agregar una columna con la conversión de mensajes a listas de palabras
# Sin eliminar las palabras vacías
movies_reviews["words"] = list(map(lambda row: row.split(), movies_reviews.review))
corpus_len = len(movies_reviews.words)

# Agregar una columna con la conversión de mensajes a listas de palabras
# Se eliminan las palabras vacías
stop_words = set(stopwords.words("english"))                  
most_common_words = nltk.FreqDist(w for wl in movies_reviews.words for w in wl)
#my_stop_words = [ w for (w,f) in most_common_words.most_common(15)]

movies_reviews["words"] = list(map(lambda row: [w for w in row.split() if not w in stop_words], 
                                   movies_reviews.review))

movies_reviews["bigrams"] = list(map(lambda row: list(ngrams(row,2)), 
                                   movies_reviews.words))

# Generar un arreglo con los valores de clasificación
Sentiments = np.array([int(x) for x in movies_reviews.sentiment])


#movies_reviews["trigrams"] = list(map(lambda row: list(ngrams(row,3)), 
#                                   movies_reviews.words))


#words_frq = nltk.FreqDist(w.lower() for wl in movies_reviews.words for w in wl
#                         ).most_common(4000)
bigrams_frq = nltk.FreqDist(w for wl in movies_reviews.bigrams for w in wl
                            ).most_common(4000)

print("empezando")       
start_time = time.time()                   
bag_idf = {}
for i, (elem, f) in zip (range(len(bigrams_frq)), bigrams_frq):
    nt = 0
    for row in movies_reviews.bigrams: 
        if elem in row:
            nt += 1
    bag_idf[elem] = (i, np.log(corpus_len / nt))

featuresets_bigrams = [
    document_features_ngrams(nltk.FreqDist(w for w in d), 
                             bag_idf) for d in movies_reviews["bigrams"]]


#trigrams_frq = nltk.FreqDist(w for wl in movies_reviews.trigrams for w in wl
#                            ).most_common(4000)
  
#featuresets_words = [
#    document_features_ngrams(d, words_frq) for d in movies_reviews["words"]]
#
#bag_dict = {}
#for i, (elem, f) in zip (range(len(bigrams_frq)), bigrams_frq):
#    bag_dict[elem] = (i, float(f)/word_bag_len)
#featuresets_bigrams = [
#    document_features_ngrams(nltk.FreqDist(d), bigrams_frq) 
#    for d in movies_reviews["bigrams"]]
#featuresets_trigrams = [
#    document_features_ngrams(nltk.FreqDist(d), trigrams_frq) 
#    for d in movies_reviews["trigrams"]]
elapsed_time = time.time() - start_time


#for i in range(100):
#    print(sum(x > 0 for x in featuresets_bigrams[i]))


bigrams_train, bigrams_test, biy_train, biy_test = train_test_split(
    featuresets_bigrams, Sentiments, test_size=0.1)

# Entrenamiento de un clasificador Multinomial Bayes ingenuo
clfM = MultinomialNB()
clfM.fit(bigrams_train, biy_train)
print(elapsed_time)    

# Pruebas del clasificador
predictions_train = clfM.predict(bigrams_train)
fails_train = sum(biy_train  != predictions_train)
print("Puntos mal clasificados en el conjunto de entrenamiento: {} de {} ({}%)\n"
      .format(fails_train, len(bigrams_train), 100*fails_train/len(bigrams_train)))
predictions_test = clfM.predict(bigrams_test)
fails_test = sum(biy_test  != predictions_test)
print("Puntos mal clasificados en el conjunto de prueba: {} de {} ({}%)\n"
      .format(fails_test, len(bigrams_test), 100*fails_test/len(bigrams_test)))