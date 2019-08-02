import torch
from torch import nn


class LayerLM(nn.Module):
    def __init__(self, last_hidden_size, vocab_size):
        super(LayerLM, self).__init__()
        self.fc = nn.Linear(last_hidden_size, vocab_size)

    def forward(self, x, *args):
        """

        :param x: (..., seq_len, last_hidden_size)
        :param args:
        :return:
        """
        with torch.no_grad():
            logits = self.get_logits(x)
            return torch.argmax(logits, dim=-1)

    def get_logits(self, x):
        return self.fc(x)
