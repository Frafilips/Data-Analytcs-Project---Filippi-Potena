import data_pre_processing 
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

x_train = data_pre_processing.x_train
y_train = data_pre_processing.y_train

x_test = data_pre_processing.x_test
y_test = data_pre_processing.y_test

x_val = data_pre_processing.x_val
y_val = data_pre_processing.y_val


#Calcolo con classifier Naive Bayes
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
#Calcolo con classifier Tree Based

