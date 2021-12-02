import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car_name"]
data = pd.read_csv('auto-mpg.data', sep="\s+", names=names)
print(data)

X, y = data.iloc[:, 1:-1], data.iloc[:, 0]

X = X.replace('?', np.NaN)
X.info()

lr = LinearRegression()
unknown_data = X[X['horsepower'].isnull() == True]
known_data = X[X['horsepower'].isnull() == False]

known_y = known_data['horsepower']
known_X = known_data.drop('horsepower', axis=1)
lr.fit(known_X, known_y)

unknown_X = unknown_data.drop('horsepower', axis=1)

print(unknown_X)

pred = lr.predict(unknown_X)
print(pred)

unknown_X['horsepower'] = pred

print(unknown_X['horsepower'])

print(unknown_X.index)

X.loc[unknown_X.index, 'horsepower'] = unknown_X['horsepower']

X.info()

X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# for name in X.columns:
#     plt.scatter(y, np.int64(X[name]))
#     plt.xlabel('mpg')
#     plt.ylabel(name)
#     plt.show()


def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * np.linalg.norm(x - c) ** 2)


def kmeans(X, k):
    kmeans_obj = KMeans(n_clusters=k, random_state=42).fit(X)
    cluster_centers_ = kmeans_obj.cluster_centers_
    stds = np.zeros(k)

    for i in range(k):
        stds[i] = np.std(np.array((X[kmeans_obj.labels_ == i])))

    return cluster_centers_, stds


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)
        # training
        for epoch in range(self.epochs):
            # giving every input to the network
            for i in range(X.shape[0]):
                # forward pass
                phi = np.array([self.rbf(np.array(X)[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = phi.T.dot(self.w) + self.b
                loss = (y[i] - F).flatten() ** 2
                print(loss)
                # print('Loss: {0:.2f}'.format(loss[0]))
                # backward pass
                error = -(y[i] - F).flatten()
                # online update
                self.w = self.w - self.lr * phi * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            phi = np.array([self.rbf(np.array(X)[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = phi.T.dot(self.w) + self.b
            y_pred.append(F)

        return np.array(y_pred)


# sample inputs and add noise

rbfnet = RBFNet(lr=1e-2, k=3, epochs=10, inferStds=False)
rbfnet.fit(X_train.to_numpy(), y_train.to_numpy())
y_pred = rbfnet.predict(X_train.to_numpy())

for a, b in zip(y, y_pred):
    print('y: {0:.2f} , y_pred: {0:.2f}'.format(a, b))
# plt.plot(X, y, '-o', label='true')
# plt.plot(X, y_pred, '-o', label='RBF-Net')
# plt.legend()
# plt.tight_layout()
# plt.show()
