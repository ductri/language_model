import unittest

import torch

from model_def.T_LM.decoder_layer_customized import DecoderLayerCustomized
from model_def.transformer import helper


class TestDecoderLayer(unittest.TestCase):

    def test_forward(self):
        decoder_layer = DecoderLayerCustomized(d_model=512, num_heads=8)
        word_input = torch.randn(size=(5, 100, 512))

        output, _ = decoder_layer(word_input, look_ahead_mask=None, target_padding_mask=None)
        self.assertListEqual(list(output.size()), [5, 100, 512])

    def test_forward_with_mask(self):
        decoder_layer = DecoderLayerCustomized(d_model=512, num_heads=8)
        word_input = torch.randn(size=(5, 100, 512))
        look_ahead_mask = helper.create_look_ahead_mask(100)
        length = torch.ones(size=(5,))*10
        target_padding_mask = helper.create_padding_mask(length, max_len=100)

        output, _ = decoder_layer(word_input, look_ahead_mask=look_ahead_mask, target_padding_mask=target_padding_mask)
        self.assertListEqual(list(output.size()), [5, 100, 512])


if __name__ == '__main__':
    unittest.main()
