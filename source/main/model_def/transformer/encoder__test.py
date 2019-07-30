import unittest

import torch
from torch import nn

from model_def.transformer.encoder import Encoder
from model_def.transformer import helper


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        word_embedding = nn.Embedding(1000, 512)
        encoder = Encoder(num_layers=6, d_model=512, num_heads=8, word_embedding=word_embedding)
        word_input = torch.randint(1000, size=(5, 100))
        output = encoder(word_input, padding_mask=None)
        self.assertListEqual(list(output.size()), [5, 100, 512])

    def test_forward_with_mask(self):
        word_embedding = nn.Embedding(1000, 512)
        encoder = Encoder(num_layers=6, d_model=512, num_heads=8, word_embedding=word_embedding)

        word_input = torch.randint(1000, size=(5, 100))
        source_seq_len = torch.randint(100, size=(5,))
        padding_mask = helper.create_padding_mask(source_seq_len, max_len=100)
        output = encoder(word_input, padding_mask=padding_mask)
        self.assertListEqual(list(output.size()), [5, 100, 512])


if __name__ == '__main__':
    unittest.main()

