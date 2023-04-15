# @Authors: Filippi Francesco - francesco.filippi4@studio.unibo.it - 0001059730, 
#           Potena Nicandro - nicandro.potena@studio.unibo.it - 0001000827
# @Date: 01/05/2023 (gg/mm/aaaa)
import numpy as np
import pandas as pd
import data_acquisition as data_acquisition
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Data Cleaning & Pre-processing
#Eliminiamo eventuali righe con valori null e duplicate
movies = data_acquisition.movies
ratings = data_acquisition.ratings
tags = data_acquisition.tags
genome_scores = data_acquisition.genome_scores
genome_tags = data_acquisition.genome_tags

#Visualizzazione rating tramite istogramma
conteggio=ratings['rating'].value_counts().sort_index()
rating=conteggio.index.array
counts=conteggio.array
fig, ax = plt.subplots()
ax.bar(rating,counts,width=0.3)
ax.ticklabel_format(axis='y', style='plain')
plt.xticks(rating)
plt.xlabel('Rating')
plt.ylabel('Occurrencies')
plt.title('Rating distribution')
#plt.show()


plt.clf()
plt.boxplot(genome_scores["relevance"])
#plt.show()

"""print(genome_scores.relevance.max())
print(genome_scores.relevance.min())"""


"""#da eliminare perchè ci sono valori null
tags = tags.drop_duplicates()
tags = tags.dropna()"""

#movies = movies.drop_duplicates()
movies = movies.dropna()

ratings.drop("timestamp",axis=1,inplace=True)
ratings = ratings.drop_duplicates()
ratings = ratings.dropna()

genome_scores = genome_scores.drop_duplicates()
genome_scores = genome_scores.dropna()

genome_tags = genome_tags.drop_duplicates()
genome_tags = genome_tags.dropna()


#one hot encoding sulla colonna dei genres per trasformare le stringhe in numeri in quanto feature qualitativa
#in modo da effettuare più facilmente le analisi

genres = movies["genres"].str.get_dummies()
movies = movies.merge(genres, on="movieId")
#vecchia colonna dei generi è stata trasformata in colonne binary, quindi non ci serve più
#inplace altrimenti non la droppa, axis indica se index (0) o colonna (1)
movies.drop("genres",axis=1,inplace=True)
#cancellazione della colonna (no genres listed)
movies.drop("(no genres listed)",axis=1,inplace=True)

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
#print("Na trovati nel dataset: ",number_of_na)

#controllo che non ci siano rating < 0 e > 5 
df_description = final_dataframe.describe().loc[["min","max"],:]
#print("Rating minimo: ", df_description.loc['min']['rating'])
#print("Rating massimo: ", df_description.loc['max']['rating'])

#ciclo che scorre tutte le colonne tranne rating 
#controllo che non ci siano valori < 0 e > 1 
check_min = (df_description.loc[['min'], df_description.columns != "rating"] < 0).sum().sum()
check_max = (df_description.loc[['max'], df_description.columns != "rating"] > 1).sum().sum()

#print("Numero di valori minori di 0 delle colonne: ", check_min)
#print("Numero di valori maggiori di 1 delle colonne : ", check_max)


#TODO colonna no genres listed da pensare di eliminare)

#Binning dei rating da 0 a 5 a step di 0.5 come indicato dal readme del dataset 
# in cui le votazioni sono date da 0 a 5 in step di 0.5

bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
binned_ratings = pd.cut(final_dataframe['rating'], bins, labels=bins[1:])
#Encoding dei rating classificati nei bin necessario per oversampling
label_encoder = LabelEncoder()
label_encoder.fit(binned_ratings)
binned_ratings = label_encoder.transform(binned_ratings)
#i valori di binned_rating vengono convertiti in int, riscalando i valori, ottenendo questa scala: [0 1 2 3 4 5 6 7]
#questo si fa perchè non si possono addestrare modelli per un problema di classificazione con numeri continui
#Sostituizione nel dataframe della colonna rating con i valori binnati e trasformati
final_dataframe['rating'] = binned_ratings
#Drop della classe con 1 solo valore per problemi con utilizzo di SMOTE
final_dataframe = final_dataframe[final_dataframe['rating'] != 0]
binned_ratings = binned_ratings[binned_ratings != 0]

#Funzioni di scaling possono essere calcolate sia prima sia dopo aver effettuato il resample perchè vanno calcolate per ogni sample(riga)
#scaling data
scaler = Normalizer(copy=False, norm='l2')
# scaler = Normalizer(copy=False, norm='l1')
# scaler = StandardScaler(copy=False)
# scaler = MinMaxScaler(copy=False)

columnsSaved = final_dataframe.columns
final_dataframe = scaler.fit_transform(final_dataframe.to_numpy())
final_dataframe[:, -1] = binned_ratings

final_dataframe = pd.DataFrame(final_dataframe, columns=columnsSaved)

#Costruzione dei set per il modello
#splitto il dataset in x e y 
df_x = final_dataframe.drop('rating',axis=1)
df_y = final_dataframe['rating']

x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size = 0.2, random_state=42
)

x_train, x_val, y_train, y_val = train_test_split (
    x_train, y_train, test_size=0.25, random_state=42
)

#Eseguo riduzione delle dimensioni provando sia con il metodo PCA sia con il metodo LDA
#PCA

"""pca = PCA(0.8)
pca.fit(x_train)
plt.clf()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Numero di componenti principali')
plt.ylabel('Varianza spiegata cumulativa')
plt.show()
plt.clf()
# traccia il grafico del gomito per selezionare il numero di componenti principali
plt.plot(range(1, x_train.shape[1]+1), np.cumsum(pca.explained_variance_ratio_), '-o')
plt.xlabel('Numero di componenti principali')
plt.ylabel('Varianza spiegata cumulativa')
plt.show()

x_train = pca.transform(x_train)
x_val = pca.transform(x_val)
x_test = pca.transform(x_test)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.show()
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
plt.show()"""


#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

x_train = lda.transform(x_train)
x_val = lda.transform(x_val)
x_test = lda.transform(x_test)
plt.clf()
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
#plt.show()
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
#plt.show()

#Bilanciamento delle classi prima di applicare SMOTE
plt.hist(label_encoder.inverse_transform(y_train.astype(int)), bins="auto")
plt.xlabel("Rating Medio")
plt.ylabel("Occorrenze")
#plt.show()

#Bilanciamento delle classi tramite oversampling
oversampler = SMOTE(k_neighbors = 6)
x_train, y_train = oversampler.fit_resample(x_train, y_train)

plt.hist(label_encoder.inverse_transform(y_train.astype(int)), bins="auto")
plt.xlabel("Rating Medio")
plt.ylabel("Occorrenze")
#plt.show()