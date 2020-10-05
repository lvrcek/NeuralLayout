import torch.nn as nn


class TerminationNetwork(nn.Module):

    def __init__(self, in_channels, out_channels=1, bias=False):
        super(TerminationNetwork, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.linear(x)