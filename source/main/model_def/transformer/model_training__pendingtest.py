import unittest

import torch
from torch import nn

from model_def.transformer.model_training import Model


class TestModel(unittest.TestCase):

    def test_forward(self):
        encoder_embedding = nn.Embedding(1000, 512)
        decoder_embedding = nn.Embedding(1001, 512)
        model = Model(num_layers=3, d_model=512, num_heads=8, encoder_embedding=encoder_embedding, decoder_embedding=decoder_embedding, dropout_rate=0.1)

        word_input = torch.randint(1000, size=(5, 80))
        target_input = torch.randint(1000, size=(5, 100))
        source_length = torch.randint(80, size=(5, ))
        target_length = torch.randint(100, size=(5, ))
        output = model(word_input, target_input, source_length, target_length)
        self.assertListEqual(list(output.size()), [5, 100, 1001])


if __name__ == '__main__':
    unittest.main()

