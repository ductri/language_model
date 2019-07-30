import unittest

import torch

from model_def.transformer.decoder_layer import DecoderLayer


class TestDecoderLayer(unittest.TestCase):

    def test_forward(self):
        decoder_layer = DecoderLayer(d_model=512, num_heads=8)
        word_input = torch.randn(size=(5, 100, 512))
        encoder_output = torch.randn(size=(5, 101, 512))

        output, _, _ = decoder_layer(word_input, encoder_output, look_ahead_mask=None, source_padding_mask=None)
        self.assertListEqual(list(output.size()), [5, 100, 512])


if __name__ == '__main__':
    unittest.main()
