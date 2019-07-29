import numpy as np

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()

        angle_rads = self.get_angles(np.arange(max_seq_len)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[np.newaxis, ...]

        embedding = torch.from_numpy(pos_encoding).float()
        self.register_buffer('embedding', embedding)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def forward(self, x, *input):
        """

        :param x: (batch, seq_len, d_model)
        :param input:
        :return:
        """
        return x + self.embedding[:, :x.size(1), :]


if __name__ == '__main__':
    pe = PositionalEncoding(100, 512)
    x = torch.zeros(10, 50, 512)
    y = pe(x)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.pcolormesh(y[0], cmap='RdBu')
    plt.show()
