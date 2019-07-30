import unittest

import torch

from model_def.transformer.encoder_layer import EncoderLayer


class TestEncoderLayer(unittest.TestCase):

    def test_forward(self):
        encoder_layer = EncoderLayer(d_model=512, num_heads=8)
        word_input = torch.randn(size=(5, 100, 512))
        output = encoder_layer(word_input, mask=None)
        self.assertListEqual(list(output.size()), [5, 100, 512])


if __name__ == '__main__':
    unittest.main()

