import unittest

import torch
from torch import nn

from model_def.transformer.decoder import Decoder
from model_def.transformer import helper


class TestDecoder(unittest.TestCase):

    def test_forward(self):
        word_embedding = nn.Embedding(1000, 512)
        decoder = Decoder(num_layers=6, d_model=512, num_heads=8, word_embedding=word_embedding)

        word_input = torch.randint(1000, size=(5, 80))
        encoder_output = torch.randn(size=(5, 100, 512))
        output = decoder(word_input, encoder_output, look_ahead_mask=None, source_padding_mask=None, target_padding_mask=None)
        self.assertListEqual(list(output.size()), [5, 80, 512])

    def test_forward_with_masks(self):
        word_embedding = nn.Embedding(1000, 512)
        decoder = Decoder(num_layers=6, d_model=512, num_heads=8, word_embedding=word_embedding)

        word_input = torch.randint(1000, size=(5, 80))
        source_seq_len = torch.randint(100, size=(5,))
        padding_mask = helper.create_padding_mask(source_seq_len, max_len=100)
        encoder_output = torch.randn(size=(5, 100, 512))
        look_ahead_mask = helper.create_look_ahead_mask(80)
        output = decoder(word_input, encoder_output, look_ahead_mask=look_ahead_mask, source_padding_mask=padding_mask, target_padding_mask=None)
        self.assertListEqual(list(output.size()), [5, 80, 512])


if __name__ == '__main__':
    unittest.main()

