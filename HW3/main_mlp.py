import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car_name"]
data = pd.read_csv('auto-mpg.data', sep="\s+", names=names)
print(data)

X, y = data.iloc[:, 1:-1], data.iloc[:, 0]

X = X.replace('?', np.NaN)
X.info()


def impute(column):
    lr = LinearRegression()

    unknown_data = X[X[column].isnull() == True]
    known_data = X[X[column].isnull() == False]

    known_y = known_data[column]
    known_X = known_data.drop(column, axis=1)

    lr.fit(known_X, known_y)

    unknown_X = unknown_data.drop(column, axis=1)
    # unknown_X

    pred = lr.predict(unknown_X)
    # pred

    unknown_X[column] = pred

    return unknown_X

unknown_X = impute('horsepower')
# unknown_X.index

X.loc[unknown_X.index, 'horsepower'] = unknown_X['horsepower']

X.info()

X = StandardScaler().fit_transform(X)

X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def evaluate(y, y_pred):
    print('mean squared error = ', mean_squared_error(y, y_pred))
    plt.scatter(y, y_pred)
    plt.xlabel('train')
    plt.ylabel('prediction')
    plt.tight_layout()
    plt.show()


def network(layers_n, perceptron_n):
    MLP = nn.Sequential()
    MLP.add_module('input', nn.Linear(7, perceptron_n))

    for i in range(layers_n):
        MLP.add_module('hidden' + str(i), nn.Linear(perceptron_n, perceptron_n))
        MLP.add_module('activation' + str(i), nn.LeakyReLU())

    MLP.add_module('output', nn.Linear(perceptron_n, 1))

    return MLP


def train(model, X, y, learning_rate=1e-6):
    loss_fn = nn.MSELoss(reduction='sum')

    for epoch in range(4):
        for x, target in zip(X, y):
            y_pred = model(x)

            # y_pred is an array with size 1 but target is just a number it leads to a warning
            loss = loss_fn(y_pred[0], target)
            if epoch % 100 == 99:
                print(epoch, loss.item())

            # Zero the gradients before running the backward pass.
            model.zero_grad()

            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

        learning_rate /= 2


X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)

model = network(2, 4).double()

start = time.time()
train(model, X_train, y_train, learning_rate=0.00001)
end = time.time()

print('Training took', end - start)

y_pred_train = model(X_train)

evaluate(y_train.detach().numpy(), y_pred_train.detach().numpy())
