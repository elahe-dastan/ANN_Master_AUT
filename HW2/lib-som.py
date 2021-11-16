from sklearn.datasets import fetch_rcv1
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn_som.som import SOM

X, y = fetch_rcv1(shuffle=True, return_X_y=True)
X_small, y_small = X[:200], y[:200]
s = SOM(m=9, n=9, dim=2)
s.fit(X_small.to_array())
p = s.predict(X_small)
