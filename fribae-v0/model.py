import torch
import torch.nn as nn


class NeuralNet(nn.Module):

    # feed-forward neural net with 2 hidden layers
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # build neural net layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax
        return out
