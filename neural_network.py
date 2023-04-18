import data_pre_processing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error,r2_score,classification_report,roc_curve, auc
import warnings
warnings.filterwarnings('always')
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import itertools


df_x=data_pre_processing.df_x
df_y=data_pre_processing.df_y

x_train = data_pre_processing.x_train
y_train = data_pre_processing.y_train

x_test = data_pre_processing.x_test
y_test = data_pre_processing.y_test

x_val = data_pre_processing.x_val
y_val = data_pre_processing.y_val

y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
y_val=y_val.to_numpy()

#Riscaling classi per utilizzare in modo corretto pytorch
y_test=y_test-1
y_train=y_train-1
y_val=y_val-1

class MovieLens(Dataset):
    def __init__(self,x,y):
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]



# Creazione del modello
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,dropout):
        super(Feedforward, self).__init__()
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, criterion, optimizer, epochs, data_loader,device):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        for data,targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass 
            y_pred = model(data)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), targets)
            
            loss_values.append(loss.item())
            #print('Epoch {} train loss: {}'.format(epoch, loss.item()))

            # Backward pass
            loss.backward()
            optimizer.step()

    return model, loss_values

def val_model(model, data_loader,best_accuracy,device,parameters,best_parameters):
    model.eval()
    y_pred = []
    y_val = []
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        y_pred.append(model(data))
        y_val.append(targets)
    y_pred = torch.stack(y_pred).squeeze().cpu()
    y_val = torch.stack(y_val).squeeze().cpu()
    y_pred = y_pred.argmax(dim=1, keepdim=True).squeeze().cpu()
    score = torch.sum((y_pred.squeeze() == y_val).float()) / y_val.shape[0]
    #print('Test score', score.numpy())
    print(classification_report(y_val, y_pred,zero_division=0))
    accuracy=accuracy_score(y_val, y_pred)
    print("accuracy: ",accuracy)
    if(accuracy>best_accuracy):
        best_accuracy=accuracy
        best_parameters=parameters

    return best_accuracy,best_parameters

def test_model(model, data_loader,device):
    model.eval()
    y_pred = []
    y_test = []
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        y_pred.append(model(data))
        y_test.append(targets)
    y_pred = torch.stack(y_pred).squeeze().cpu()
    y_test = torch.stack(y_test).squeeze().cpu()
    y_pred = y_pred.argmax(dim=1, keepdim=True).squeeze().cpu()
    score = torch.sum((y_pred.squeeze() == y_test).float()) / y_test.shape[0]
    #print('Test score', score.numpy())
    print("Classification report")
    print(classification_report(y_test, y_pred,zero_division=0))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
    print(conf_matrix)
    accuracy=accuracy_score(y_test, y_pred)
    print("Accuracy: ",accuracy)

    
    # Calcolo della curva ROC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_test))):
        
        fpr[i], tpr[i], _ = roc_curve(y_test==i, y_pred==i)
        
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calcolo della media dei valori di AUC
    mean_auc = np.mean(list(roc_auc.values()))
    print("AUC mean: ",mean_auc)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    best_accuracy=0
    """hidden_sizes = [64,128]
    nums_epochs = [50,100]
    batch = [16,32,64]
    learning_rate = [0.01,0.001]
    momentum=[0.9,0.95]
    dropout=[0.1,0.2]"""
    #miglior parametri con LDA: hs:128;ne:100;b:64;lr:0.01;momentum=0.95;do:0.1; 
    #miglior parametri con PCA: hs:128;ne:50;b:64;lr:0.01;momentum=0.95;do:0.2; 
    hidden_sizes = [128]
    nums_epochs = [100]
    batch = [64]
    learning_rate = [0.01]
    momentum=[0.95]
    dropout=[0.1]
    
    best_parameters=[]
    hyperparameters = itertools.product(hidden_sizes, nums_epochs, batch,learning_rate,momentum,dropout)

    datasetTrain=MovieLens(x_train, y_train)
    datasetTest=MovieLens(x_test, y_test)
    datasetVal=MovieLens(x_val, y_val)

    val_loader=DataLoader(datasetVal, batch_size=1, shuffle=True)
    test_loader=DataLoader(datasetTest, batch_size=1, shuffle=True)
    
    for hidden_size, num_epochs, batch, learning_rate, momentum,dropout in hyperparameters:
        parameters=[]
        parameters.append(hidden_size)
        parameters.append(num_epochs)
        parameters.append(batch)
        parameters.append(learning_rate)
        parameters.append(momentum)
        parameters.append(dropout)

        torch.manual_seed(42)
        np.random.seed(42)
        #torch.use_deterministic_algorithms(True)
        train_loader=DataLoader(datasetTrain, batch_size=batch, shuffle=True,drop_last=True)
        model = Feedforward(x_train.shape[1], hidden_size, datasetTrain.num_classes,dropout)
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss() #Softmax and NNLL, does not require one-hot encoding of labels
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)

        #test_model(model,val_loader, device)
        model, loss_values = train_model(model, criterion, optimizer, num_epochs, train_loader, device)
        print("Hiddensize: ",hidden_size,"; Num epochs: ",num_epochs,"; Batch: ",batch,"; Learning rate: ",learning_rate,"; Momentum: ",momentum,"; Dropout: ",dropout)
        best_accuracy,best_parameters=val_model(model, val_loader,best_accuracy, device,parameters,best_parameters)
        
        plt.clf()
        plt.plot(loss_values)
        plt.title("Number of epochs: {}".format(num_epochs))
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

    print("Best accuracy: ",best_accuracy)
    #Training with best parameters and testing the model
    best_hidden_size=best_parameters[0]
    best_num_epochs=best_parameters[1]
    best_batch=best_parameters[2]
    best_learning_rate=best_parameters[3]
    best_momentum=best_parameters[4]
    best_dropout=best_parameters[5]
    print("Best parameters: hidden_size:",best_hidden_size,"; num_epochs: ",best_num_epochs,"; batch: ",best_batch,"; learning_rate: ",best_learning_rate,": momentum: ",best_momentum,"; dropout: ",best_dropout)
    train_loader=DataLoader(datasetTrain, batch_size=best_batch, shuffle=True,drop_last=True)
    model = Feedforward(x_train.shape[1], best_hidden_size, datasetTrain.num_classes,best_dropout)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss() #Softmax and NNLL, does not require one-hot encoding of labels
    optimizer = torch.optim.SGD(model.parameters(), lr=best_learning_rate,momentum=best_momentum)
    model, loss_values = train_model(model, criterion, optimizer, best_num_epochs, train_loader, device)

    #test 
    plt.clf()
    plt.plot(loss_values)
    plt.title("Number of epochs: {}".format(best_num_epochs))
    #plt.show()
    test_model(model, test_loader, device)

    """
    #Salvataggio modello
    print("Salvataggio modello")
    torch.save(model.state_dict(), 'nn_model.pt')

    # Carica il modello salvato
    print("Carico il modello")
    model = Feedforward(x_train.shape[1], best_hidden_size, datasetTrain.num_classes,best_dropout)
    model.to(device)
    model.load_state_dict(torch.load('nn_model.pt'))
    
    # Utilizza il modello per fare predizioni sui tuoi dati di test
    test_model(model, test_loader, device)"""