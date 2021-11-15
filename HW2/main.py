from sklearn.datasets import fetch_rcv1
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

X, y = fetch_rcv1(shuffle=True, return_X_y=True)


class SOM:
    def __init__(self, map_size, lr=0.0001):
        """
        :param map_size: [map_w, map_h, f]
        """
        self.map = np.random.random(size=map_size)
        self.lr0 = lr
        self.lr = lr

        self.R0 = map_size[0] // 2
        self.R = self.R0

    def find_winner(self, x):
        repeated_x = np.tile(x, [self.map.shape[0], self.map.shape[1], 1])
        dists = np.sum((self.map - repeated_x) ** 2, axis=2)

        winner = np.unravel_index(np.argmin(dists), dists.shape)

        return winner

    def get_NS(self, winner):
        NS = np.zeros(shape=(self.map.shape[0], self.map.shape[1]))

        iw, jw = winner[0], winner[1]

        NS[iw, jw] = 1

        for r in range(1, int(self.R)):
            if iw - r >= 0:
                NS[iw - r, jw] = 1 / r   # NS for winner should be one and for it's nearest neighbor should be 1/2
            if iw + r < self.map.shape[0]:
                NS[iw + r, jw] = 1 / r

            if jw - r >= 0:
                NS[iw, jw - r] = 1 / r
            if jw + r < self.map.shape[1]:
                NS[iw, jw + r] = 1 / r

        return NS

    def update_weights(self, x, n_strength):
        NS = np.tile(n_strength, [self.map.shape[2], 1, 1]).transpose()  # ERROR

        repeated_x = np.tile(x, [self.map.shape[0], self.map.shape[1], 1])
        Delta = self.map - repeated_x

        self.map += self.lr * np.multiply(NS, Delta)

    def train(self, X, epochs=1000, error_threshold=10 ** -20):
        Js = []
        for epoch in range(epochs):
            prev_map = np.copy(self.map)
            for row in range(X.shape[0]):
                x = X[row, :]
                x = x.toarray()
                winner = self.find_winner(x)  # winner = [5, 23]
                neighbors = self.get_NS(winner)
                self.update_weights(x, neighbors)

            self.lr = self.lr0 * (1 - epoch / epochs)
            self.R = self.R0 * (1 - epoch / epochs)

            Js.append(np.linalg.norm(prev_map - self.map))

            #             if epoch % 100 == 0:
            print("Iteration: %d, LR: %f, R: %f, J: %f" % (epoch, self.lr, self.R, Js[-1]))

            if Js[-1] < error_threshold:
                print("MIN CHANGE")
                break

        return Js


X_small = X[:1000]

som_net = SOM(map_size=[9, 9, X_small.shape[1]])
Js = som_net.train(X_small, epochs=10)

plt.plot(Js)
plt.show()
