# @Authors: Filippi Francesco - francesco.filippi4@studio.unibo.it - 0001059730, 
#           Potena Nicandro - nicandro.potena@studio.unibo.it - 0001000827
# @Date: 01/05/2023 (gg/mm/aaaa)
import numpy as np
import pandas as pd
import data_acquisition as data_acquisition
from sklearn.preprocessing import LabelEncoder

#Data Cleaning & Pre-processing
#Eliminiamo eventuali righe con valori null e duplicate
movies = data_acquisition.movies
ratings = data_acquisition.ratings
tags = data_acquisition.tags
genome_scores = data_acquisition.genome_scores
genome_tags = data_acquisition.genome_tags


"""movies = movies.drop_duplicates()
movies = movies.dropna()

ratings = ratings.drop_duplicates()
ratings = ratings.dropna()

#da eliminare perchè ci sono valori null
tags = tags.drop_duplicates()
tags = tags.dropna()

genome_scores = genome_scores.drop_duplicates()
genome_scores = genome_scores.dropna()

genome_tags = genome_tags.drop_duplicates()
genome_tags = genome_tags.dropna()"""

#one hot encoding sulla colonna dei genres per trasformare le stringhe in numeri in quanto feature qualitativa
#in modo da effettuare più facilmente le analisi
movies = data_acquisition.movies
genres = movies["genres"].str.get_dummies()
movies = movies.merge(genres, on="movieId")
#vecchia colonna dei generi è stata trasformata in colonne binary, quindi non ci serve più
#inplace altrimenti non la droppa, axis indica se index (0) o colonna (1)
movies.drop("genres",axis=1,inplace=True)

#Aggiungiamo i genomi
genome_scores = genome_scores.merge(genome_tags, on="tagId")

#ruotiamo le righe in colonne per attribuire una relevance per ogni genoma ad ogni movieId
genome_scores = genome_scores.pivot(
        index="movieId", columns="tag", values="relevance")

#calcoliamo la media dei ratings 
mean_ratings = ratings.groupby("movieId")["rating"].mean()
#print("MEAN -> ", len(mean_ratings))
#print("GENOMES -> ", len(genome_scores))
#print("MOVIES -> ", len(movies))

#Effettuiamo merge per creare il dataframe finale
#le informazioni sui genomes corrispondono solo a 13816 movies sui 62423, quando viene effettuata l'operazione di merge sul movieId vengono scartete le informazioni dei quasi 50000 mancanti
final_dataframe = movies.merge(genome_scores, on="movieId")
final_dataframe = final_dataframe.merge(mean_ratings, on="movieId")


#verifico integrità dei dati
#controllo se ci sono NA
number_of_na = final_dataframe.isna().sum().sum()
print("Na trovati nel dataset: ",number_of_na)

#controllo che non ci siano rating < 0 e > 5 
df_description = final_dataframe.describe().loc[["min","max"],:]
print("Rating minimo: ", df_description.loc['min']['rating'])
print("Rating massimo: ", df_description.loc['max']['rating'])

#ciclo che scorre tutte le colonne tranne rating 
#controllo che non ci siano valori < 0 e > 1 
check_min = (df_description.loc[['min'], df_description.columns != "rating"] < 0).sum().sum()
check_max = (df_description.loc[['min'], df_description.columns != "rating"] > 1).sum().sum()

print("Numero di valori minori di 0 delle colonne: ", check_min)
print("Numero di valori maggiori di 1 delle colonne : ", check_max)


#TODO colonna no genres listed da pensare di eliminare)

#Binning dei rating da 0 a 5 a step di 0.5 come indicato dal readme del dataset 
# in cui le votazioni sono date da 0 a 5 in step di 0.5

bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

final_dataframe['binned_ratings'] = pd.cut(final_dataframe['rating'], bins, labels=bins[1:])
print(final_dataframe)

"""binned_ratings = pd.cut(
        final_dataframe['rating'], bins=bins, labels=bins[1:])
print(binned_ratings) 
label_encoder = LabelEncoder()
label_encoder.fit(binned_ratings)
binned_ratings = label_encoder.transform(binned_ratings) 
print(binned_ratings) 
print(label_encoder)"""

