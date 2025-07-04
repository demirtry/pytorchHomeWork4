import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, hidden_sizes: list[int], input_dim: int = 28*28, num_classes: int=10):
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, num_classes)]
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
