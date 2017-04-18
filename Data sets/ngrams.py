# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:00:34 2017

@author: rsotoc
"""

import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords 
from nltk.util import ngrams

from bs4 import BeautifulSoup
#from sklearn.naive_bayes import BernoulliNB, MultinomialNB

def document_features_ngrams(document, global_features): 
    document_ngrams = nltk.FreqDist(w for wl in movies_reviews.bigrams for w in wl) 
    features = [0] * len(global_features)
    for index, (elem, f) in enumerate(global_features):
        tf = np.log(document_ngrams.freq(elem))
        idf = np.log(1 / f)
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

# Agregar una columna con la conversión de mensajes a listas de palabras
# Se eliminan las palabras vacías
#stops = set(stopwords.words("english"))                  

movies_reviews["words"] = list(map(lambda row: [w for w in row.split() if not w in stops], 
                                   movies_reviews.review))

#movies_reviews["bigrams"] = list(map(lambda row: list(ngrams(row,2)), 
#                                   movies_reviews.words))

movies_reviews["trigrams"] = list(map(lambda row: list(ngrams(row,3)), 
                                   movies_reviews.words))


#words_frq = nltk.FreqDist(w.lower() for wl in movies_reviews.words for w in wl
#                         ).most_common(4000)
#bigrams_frq = nltk.FreqDist(w for wl in movies_reviews.bigrams for w in wl
#                           ).most_common(4000)
trigrams_frq = nltk.FreqDist(w for wl in movies_reviews.trigrams for w in wl
                            ).most_common(4000)
                            
#featuresets_words = [
#    document_features_ngrams(d, words_frq) for d in movies_reviews["words"]]
#
#featuresets_bigrams = [
#    document_features_ngrams(d, bigrams_frq) for d in movies_reviews["bigrams"]]

featuresets_trigrams = [
    document_features_ngrams(d, trigrams_frq) for d in movies_reviews["trigrams"]]

#print(featuresets_words[:10])
#print(featuresets_bigrams[:10])
print(featuresets_trigrams[:10])