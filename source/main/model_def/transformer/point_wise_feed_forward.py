import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super(PointWiseFeedForward, self).__init__()
        self.dense1 = nn.Linear(d_model, 2048)
        self.dense2 = nn.Linear(2048, d_model)

    def forward(self, input, *args):
        """

        :param input: (batch, seq_len, d_model)
        :param args:
        :return:
        """
        temp = self.dense1(input)
        temp = F.relu(temp)
        output = self.dense2(temp)
        return output
