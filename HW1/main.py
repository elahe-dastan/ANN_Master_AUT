import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from NN import NeuralNetwork


def plot_dataset():
    fig, axs = plt.subplots(7, 3, figsize=(10, 10))
    fig.tight_layout(pad=3.0)
    cdict = {1: 'red', 2: 'blue', 3: 'green'}
    column = 0
    row = 0
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(1, 4):
                ix = np.where(train_dataset[:, 7] == k)
                axs[row, column].set_xlabel(i)
                axs[row, column].set_ylabel(j)
                axs[row, column].scatter(train_dataset[ix, i], train_dataset[ix, j], c=cdict[k])

            column += 1
            if column == 3:
                column = 0
                row += 1

    plt.savefig("visualize_data.png")
    plt.show()


def show_graph():
    writer = SummaryWriter()
    writer.add_graph(model, train_dataset.double())
    writer.close()


if __name__ == '__main__':
    data = pd.read_csv('dataset/seeds_dataset.txt', sep="\s+", header=None)

    # As it's mentioned in the question file, the code should be in tensorflow/pytorch so I change the pandas.DataFrame
    # to Tensor
    # Before any investigation we should split the data
    train_dataset, test_dataset = train_test_split(data, test_size=0.30)
    test_dataset, validation_dataset = train_test_split(test_dataset, test_size=0.33)

    train_dataset, test_dataset, validation_dataset = train_dataset.to_numpy(), test_dataset.to_numpy(), validation_dataset.to_numpy()

    # Q1 plotting the dataset
    # plot_dataset()

    train_dataset, train_target = torch.tensor(train_dataset[:, :7]), torch.tensor(train_dataset[:, 7])
    test_dataset, test_target = torch.tensor(test_dataset[:, :7]), torch.tensor(test_dataset[:, 7])
    validation_dataset, validation_target = torch.tensor(validation_dataset[:, :7]), torch.tensor(
        validation_dataset[:, 7])

    # Build a Neural Network model
    model = NeuralNetwork().double()

    # Show graph
    # show_graph()

    model.train_model(train_dataset, train_target.detach().clone(), 20000)

    acc = model.evaluate_model(train_dataset, train_target)
    print('Train Dataset Accuracy: %.3f' % acc)

    acc = model.evaluate_model(validation_dataset, validation_target)
    print('Validation Dataset Accuracy: %.3f' % acc)
