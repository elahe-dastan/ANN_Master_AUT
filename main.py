import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('dataset/seeds_dataset.txt', sep="\s+", header=None)

    # As it's mentioned in the question file, the code should be in tensorflow/pytorch so I change the pandas.DataFrame
    # to Tensor
    # Before any investigation we should split the data
    train_dataset, test_dataset = train_test_split(data, test_size=0.30)
    test_dataset, validation_dataset = train_test_split(test_dataset, test_size=0.33)

    train_dataset, test_dataset, validation_dataset = train_dataset.to_numpy(), test_dataset.to_numpy(), validation_dataset.to_numpy()

    # Q1 plotting the dataset
    fig, axs = plt.subplots(7, 3, figsize=(5, 5))
    cdict = {1: 'red', 2: 'blue', 3: 'green'}
    column = 0
    row = 0
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(1, 4):
                ix = np.where(train_dataset[:, 7] == k)
                axs[row, column].scatter(train_dataset[ix, i], train_dataset[ix, j], c=cdict[k])
            column += 1
            if column == 3:
                column = 0
                row += 1

    plt.show()
