from sklearn.datasets import fetch_rcv1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

X, y = fetch_rcv1(shuffle=True, return_X_y=True)


class SOM:
    def __init__(self, map_size, lr=0.2):
        """
        :param map_size: [map_w, map_h, f]
        """
        self.map = np.random.random(size=map_size)
        self.lr0 = lr
        self.lr = lr

        self.R0 = map_size[0]  # initial R is the half of the width of the map
        self.R = self.R0

        self.scores = np.zeros(shape=(self.map.shape[0], self.map.shape[1], 3))  # scores is used for visualization

    # returns the index of the winner neuron like (3, 4)
    def find_winner(self, x):
        repeated_x = np.tile(x, [self.map.shape[0], self.map.shape[1], 1])
        dists = np.sum((self.map - repeated_x) ** 2, axis=2)

        winner = np.unravel_index(np.argmin(dists), dists.shape)

        return winner

    # returns a weight for each neuron based on how close it is to the winner. The closer the bigger
    def get_NS(self, winner):
        NS = np.zeros(shape=(self.map.shape[0], self.map.shape[1]))

        iw, jw = winner[0], winner[1]

        NS[iw, jw] = 1

        for r in range(1, int(self.R)):
            if iw - r >= 0:
                NS[iw - r, jw] = 1 / (1 + r)
            if iw + r < self.map.shape[0]:
                NS[iw + r, jw] = 1 / (1 + r)

            if jw - r >= 0:
                NS[iw, jw - r] = 1 / (1 + r)
            if jw + r < self.map.shape[1]:
                NS[iw, jw + r] = 1 / (1 + r)

        return NS

    def update_weights(self, x, n_strength):
        NS = np.tile(n_strength, [self.map.shape[2], 1, 1]).transpose()  # ERROR

        repeated_x = np.tile(x, [self.map.shape[0], self.map.shape[1], 1])
        Delta = repeated_x - self.map

        self.map += self.lr * np.multiply(NS, Delta)

    def train(self, X, epochs=1000, error_threshold=10 ** -20):
        Js = []
        for epoch in range(epochs):
            prev_map = np.copy(self.map)
            for row in range(X.shape[0]):
                x = X[row, :]
                # x = x.toarray()
                winner = self.find_winner(x)  # winner = [5, 23]
                neighbors = self.get_NS(winner)
                self.update_weights(x, neighbors)

            self.lr = self.lr0 * (1 - epoch / epochs)
            self.R = self.R0 * (1 - epoch / epochs)

            Js.append(np.linalg.norm(prev_map - self.map))

            print("Iteration: %d, LR: %f, R: %f, J: %f" % (epoch, self.lr, self.R, Js[-1]))

            if Js[-1] < error_threshold:
                print("MIN CHANGE")
                break

        return Js

    def visualize(self, X, y):
        for i, (x, label) in enumerate(zip(X, y)):
            # x = x.toarray()
            winner = self.find_winner(x)
            iw, jw = winner[0], winner[1]

            # self.scores[iw, jw] += label.toarray()[0]
            self.scores[iw, jw] += label

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                norm = np.linalg.norm(self.scores[i, j])
                if norm == 0:
                    continue
                self.scores[i, j] = self.scores[i, j] / norm

        plt.imshow(self.scores)
        plt.show()

    def purity(self, X, y):
        map = np.zeros(shape=(self.map.shape[0], self.map.shape[1], y.shape[1]))
        for i, (x, label) in enumerate(zip(X, y)):
            # x = x.toarray()
            winner = self.find_winner(x)
            iw, jw = winner[0], winner[1]
            c = np.unravel_index(np.argmax(label), label.shape)
            map[iw, jw, c] += 1

        sigma = 0
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                sigma += map[i, j].max()

        p = sigma / X.shape[0]
        return p


# _, X, _, y = train_test_split(X, y, test_size=0.1, random_state=42)

svd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
X_new = svd.fit_transform(X)

svd = TruncatedSVD(n_components=3, n_iter=5, random_state=42)
y_new = svd.fit_transform(y)
y_new += abs(y_new.min())

# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())
# print(svd.singular_values_)

som_net = SOM(map_size=[12, 12, X_new.shape[1]])
Js = som_net.train(X_new, epochs=6)

plt.plot(Js)
plt.show()

som_net.visualize(X_new, y_new)

p = som_net.purity(X_new, y_new)
print("purity = ", p)
