import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # TODO: last layer of encoder has no activation function
        sequence = []
        layers = [768, 1024, 512, 256, 32, 5]
        for i, o in zip(layers,layers[1:-1]):
            sequence.append(nn.Linear(i, o))
            sequence.append(nn.ReLU())
        sequence.append(nn.Linear(layers[-2], layers[-1]))
        self.classifier = nn.Sequential(*sequence)
 

    def forward(self, x):
        return self.classifier(x)