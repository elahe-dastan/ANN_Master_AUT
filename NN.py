import torch
from torch import nn
from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # torch.argmax(logits, dim=1)
        return logits

    def train_model(self, train_dl, targets, epochs):
        # define the optimization
        self.train()
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in range(epochs):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = self(train_dl)
            # calculate loss
            print(yhat.shape)
            print(targets.shape)
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

    def evaluate_model(self, test_dl, targets):
        self.train(False)
        predictions, actuals = list(), list()
        for i, (inputs, target) in enumerate(zip(test_dl, targets)):
            # evaluate the model on the test set
            yhat = self(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = target.numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc
