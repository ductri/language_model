import unittest

import torch

from model_def.transformer import helper


class TestEncoderLayer(unittest.TestCase):

    def test_create_padding_mask(self):
        source_seq_len = torch.Tensor([2, 5])
        mask = helper.create_padding_mask(source_seq_len=source_seq_len, max_len=7)
        expected_mask = torch.Tensor([[0, 0, 1, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 1, 1]])
        expected_mask = expected_mask[:, None, None, :]
        self.assertEqual((mask != expected_mask).sum(), 0)

    def test_create_look_ahead_mask(self):
        length = 5
        mask = helper.create_look_ahead_mask(length)
        expected_mask = torch.Tensor(
            [[0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]],
        )
        expected_mask = expected_mask[None, None, :, :]
        self.assertEqual((mask != expected_mask).sum(), 0)


if __name__ == '__main__':
    unittest.main()

