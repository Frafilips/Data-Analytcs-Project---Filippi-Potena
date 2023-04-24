# @Authors: Filippi Francesco - francesco.filippi4@studio.unibo.it - 0001059730, 
#           Potena Nicandro - nicandro.potena@studio.unibo.it - 0001000827
# @Date: 01/05/2023 (gg/mm/aaaa)
import numpy as np
import pandas as pd

__CSV_FOLDER = '../ml-25m'

#Leggo dai file CSV i dati e creo un DataFrame per ogni file
movies = pd.read_csv(__CSV_FOLDER+"/movies.csv", index_col="movieId", usecols=["movieId","genres"])
ratings = pd.read_csv(__CSV_FOLDER+"/ratings.csv")
genome_scores = pd.read_csv(__CSV_FOLDER+"/genome-scores.csv")
genome_tags = pd.read_csv(__CSV_FOLDER+"/genome-tags.csv")

#Ignoriamo i file links e tags perchè inutili ai fini della nostra analisi