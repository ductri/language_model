import unittest

import torch
from torch import nn

from model_def.T_LM.model import Model


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1)
        seq_len = torch.randint(100, size=(5,))
        x = torch.randint(1000, size=(5, 100))
        output = model(x, seq_len)
        self.assertListEqual(list(output.size()), [5, 100])


if __name__ == '__main__':
    unittest.main()

