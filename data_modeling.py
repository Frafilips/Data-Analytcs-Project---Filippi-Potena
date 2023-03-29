import data_pre_processing 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error

x_train = data_pre_processing.x_train
y_train = data_pre_processing.y_train

x_test = data_pre_processing.x_test
y_test = data_pre_processing.y_test

x_val = data_pre_processing.x_val
y_val = data_pre_processing.y_val

def naybeBayes():#Calcolo con classifier Naive Bayes

    #inizializzazione classificatore
    nb_classifier = GaussianNB()
    #addestro il classificatore
    nb_classifier.fit(x_train,y_train)
    #effettuo prediction
    prediction = nb_classifier.predict(x_test)
    #differenze from y_test e prediction
    # creazione della matrice di confusione
    conf_matrix = confusion_matrix(y_test, prediction)
    print(accuracy_score(y_test,prediction))
    print(conf_matrix)
    mse = mean_squared_error(y_test, prediction)
    print('MSE:', mse)



#Calcolo con classifier Tree Based

#Calcolo con classifier SVM

def knn():#Calcolo con classifier KNN
    mse_values = []
    best_k = None
    best_mse = float('inf')
    for k in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)

        prediction = knn.predict(x_test)

        # calcolo del MSE
        mse = mean_squared_error(y_test, prediction)

        # confronto con il miglior valore di MSE finora
        if mse < best_mse:
            best_mse = mse
            best_k = k

    # stampa del risultato migliore
    print("Il miglior valore di K è", best_k, "con un RMSE di", best_mse)
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x_train,y_train)
    prediction = best_knn.predict(x_test)
    conf_matrix = confusion_matrix(y_test, prediction)
    print("accuracy score: ",accuracy_score(y_test,prediction))
    print(conf_matrix)

#Calcolo con classifier MLP

if __name__ == "__main__":
    #naybeBayes()
    knn()