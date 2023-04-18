import data_pre_processing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error,r2_score,roc_curve, auc,classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

df_x=data_pre_processing.df_x
df_y=data_pre_processing.df_y

x_train = data_pre_processing.x_train
y_train = data_pre_processing.y_train

x_test = data_pre_processing.x_test
y_test = data_pre_processing.y_test

x_val = data_pre_processing.x_val
y_val = data_pre_processing.y_val

def naybeBayes():#Calcolo con classifier Naive Bayes
    print("NAYVE@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #inizializzazione classificatore
    nb_classifier = GaussianNB()
    #addestro il classificatore
    nb_classifier.fit(x_train,y_train)
    #effettuo prediction
    prediction = nb_classifier.predict(x_test)
    #differenze from y_test e prediction
    #print(y_test.value_counts())
    predictionSeries = pd.Series(prediction)
    #print(predictionSeries.value_counts())
    predictionSeries.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    df = pd.concat([predictionSeries, y_test],axis=1)
    df.columns = [ 'predicted rating', 'real rating' ]

    # creazione della matrice di confusione
    conf_matrix = confusion_matrix(y_test, prediction)
    print("Classification report: ",classification_report(y_test, prediction,zero_division=0))
    print("Accuracy: ",accuracy_score(y_test,prediction))
    print(conf_matrix)

    # Calcolo della curva ROC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_test))):
        
        fpr[i], tpr[i], _ = roc_curve(y_test==i+1, prediction==i+1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcolo della media dei valori di AUC
    mean_auc = np.mean(list(roc_auc.values()))
    print("AUC mean: ",mean_auc)


def decisionTree():#Calcolo con classifier Tree Based
    print("DECISION TREE@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    clf = DecisionTreeClassifier()

    # Definire i parametri da testare
    parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}

    # Cerca i migliori valori per i parametri di criterion, maxdepth e ccp_alpha utilizzando GridSearchCV
    grid_search = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Ottenere il modello con i migliori parametri
    best_model = grid_search.best_estimator_

    # Calcolare la precisione del modello sul set di test
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Visualizzare i risultati
    print('Migliori parametri:', grid_search.best_params_)
    print('Accuratezza:', accuracy)
    print("Classification report: ",classification_report(y_test, y_pred,zero_division=0))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Calcolo della curva ROC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_test))):
        
        fpr[i], tpr[i], _ = roc_curve(y_test==i+1, y_pred==i+1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcolo della media dei valori di AUC
    mean_auc = np.mean(list(roc_auc.values()))
    print("AUC mean: ",mean_auc)


def svc():#Calcolo con classifier SVM
    print("SVC@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    # Creare un oggetto SVM Classifier
    svc = SVC()

    # Creare un oggetto GridSearchCV e addestrare il modello con i diversi valori di C e del kernel
    clf = GridSearchCV(svc, params, cv=5)
    clf.fit(x_train, y_train)

    # Stampare i risultati della ricerca a griglia
    print("Miglior valore di C:", clf.best_params_['C'])
    print("Miglior kernel:", clf.best_params_['kernel'])

    best_svc=SVC(kernel='rbf',C=10)
    best_svc.fit(x_train, y_train)
    prediction = best_svc.predict(x_test)
    conf_matrix = confusion_matrix(y_test, prediction)
    print("Classification report: ",classification_report(y_test, prediction,zero_division=0))
    print("accuracy score: ",accuracy_score(y_test,prediction))
    print(conf_matrix)

    # Calcolo della curva ROC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_test))):
        
        fpr[i], tpr[i], _ = roc_curve(y_test==i+1, prediction==i+1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcolo della media dei valori di AUC
    mean_auc = np.mean(list(roc_auc.values()))
    print("AUC mean: ",mean_auc)

    

def knn():#Calcolo con classifier KNN
    print("KNN@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    mse_values = []
    best_k = None
    best_accuracy = float('inf')
    for k in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)

        prediction = knn.predict(x_val)

        accuracy = accuracy_score(y_val, prediction)

        # confronto con il miglior valore di MSE finora
        if accuracy < best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # stampa del risultato migliore
    print("Il miglior valore di K è", best_k, "con un accuracy di", best_accuracy)
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x_train,y_train)
    prediction = best_knn.predict(x_test)
    conf_matrix = confusion_matrix(y_test, prediction)
    print("Classification report: ",classification_report(y_test, prediction,zero_division=0))
    print("accuracy score: ",accuracy_score(y_test,prediction))
    print(conf_matrix)

    # Calcolo della curva ROC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_test))):
        
        fpr[i], tpr[i], _ = roc_curve(y_test==i+1, prediction==i+1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcolo della media dei valori di AUC
    mean_auc = np.mean(list(roc_auc.values()))
    print("AUC mean: ",mean_auc)


if __name__ == "__main__":
    naybeBayes()
    decisionTree()
    svc()
    knn()