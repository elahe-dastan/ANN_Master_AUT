from torch import nn
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
        criterion = CrossEntropyLoss().double()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        targets -= 1
        # enumerate epochs
        for epoch in range(epochs):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = self(train_dl)
            # calculate loss
            loss = criterion(yhat, targets.long())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

    def evaluate_model(self, test_dl, targets):
        self.train(False)
        # evaluate the model on the test set
        yhat = self(test_dl)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        targets -= 1
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # calculate accuracy
        acc = accuracy_score(targets, yhat)
        return acc
