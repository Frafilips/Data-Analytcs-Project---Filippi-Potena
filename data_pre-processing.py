# @Authors: Filippi Francesco - francesco.filippi4@studio.unibo.it - 0001059730, 
#           Potena Nicandro - nicandro.potena@studio.unibo.it - 0001000827
# @Date: 01/05/2023 (gg/mm/aaaa)
import numpy as np
import pandas as pd
import data_acquisition as data_acquisition
from sklearn.preprocessing import OneHotEncoder

#Data Cleaning & Pre-processing
#Eliminiamo eventuali righe con valori null e duplicate
movies = data_acquisition.movies
ratings = data_acquisition.ratings
tags = data_acquisition.tags
genome_scores = data_acquisition.genome_scores
genome_tags = data_acquisition.genome_tags

print(len(movies))
movies = movies.drop_duplicates()
movies = movies.dropna()
print(len(movies))

print(len(ratings))
ratings = ratings.drop_duplicates()
ratings = ratings.dropna()
print(len(ratings))

#da eliminare perchè ci sono valori null
print(len(tags))
tags = tags.drop_duplicates()
tags = tags.dropna()
print(len(tags))

print(len(genome_scores))
genome_scores = genome_scores.drop_duplicates()
genome_scores = genome_scores.dropna()
print(len(genome_scores))

print(len(genome_tags))
genome_tags = genome_tags.drop_duplicates()
genome_tags = genome_tags.dropna()
print(len(genome_tags))

#one hot encoding sulla colonna dei genres per trasformare le stringhe in numeri in quanto feature qualitativa
#in modo da effettuare più facilmente le analisi
movies = data_acquisition.movies
enc = OneHotEncoder()

movies = enc.fit_transform(movies)


