import data_pre_processing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error,r2_score

import torch
from torch.utils.data import Dataset, DataLoader, Subset

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
    def __init__(self, input_size, hidden_size, num_classes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

def train_model(model, criterion, optimizer, epochs, data_loader):
    model.train()
    loss_values = []
    for epoch in range(epochs):
        for data,targets in data_loader:

            optimizer.zero_grad()

            # Forward pass 
            y_pred = model(data)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), targets)
            
            loss_values.append(loss.item())
            print('Epoch {} train loss: {}'.format(epoch, loss.item()))

            # Backward pass
            loss.backward()
            optimizer.step()

    return model, loss_values

def test_model(model, data_loader):
    model.eval()
    y_pred = []
    y_test = []
    for data, targets in data_loader:
        y_pred.append(model(data))
        y_test.append(targets)
    y_pred = torch.stack(y_pred).squeeze()
    y_test = torch.stack(y_test).squeeze()
    y_pred = y_pred.argmax(dim=1, keepdim=True)
    score = torch.sum((y_pred.squeeze() == y_test).float()) / y_test.shape[0]
    print('Test score', score.numpy())


if __name__ == "__main__":
    
    hidden_size = 32
    num_epochs = 10
    learning_rate = 0.001
    batch = 32

    datasetTrain=MovieLens(x_train, y_train)
    datasetTest=MovieLens(x_test, y_test)
    datasetVal=MovieLens(x_val, y_val)

    train_loader=DataLoader(datasetTrain, batch_size=batch, shuffle=True,drop_last=True)
    val_loader=DataLoader(datasetVal, batch_size=1, shuffle=True)
    
    

    model = Feedforward(x_train.shape[1], hidden_size, datasetTrain.num_classes)
    criterion = torch.nn.CrossEntropyLoss() #Softmax and NNLL, does not require one-hot encoding of labels
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    test_model(model,val_loader)
    model, loss_values = train_model(model, criterion, optimizer, num_epochs, train_loader)
    test_model(model, val_loader)
    plt.clf()
    plt.plot(loss_values)
    plt.title("Number of epochs: {}".format(num_epochs))
    plt.show()