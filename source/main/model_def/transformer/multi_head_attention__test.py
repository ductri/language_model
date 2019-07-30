import unittest

import torch

from model_def.transformer.multi_head_attention import MultiHeadAttention
from model_def.transformer import helper


class TestMultiHeadAttention(unittest.TestCase):
    def test_scale_dot_product(self):
        q = torch.randn(size=(5, 50, 512))
        k = torch.randn(size=(5, 50, 512))
        v = torch.randn(size=(5, 50, 512))
        output, attn_weights = MultiHeadAttention.scaled_dot_product_attention(q, k, v, mask=None)
        self.assertListEqual(list(output.size()), [5, 50, 512])
        self.assertListEqual(list(attn_weights.size()), [5, 50, 50])

    def test_forward(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        q = torch.randn(size=(5, 50, 512))
        k = torch.randn(size=(5, 60, 512))
        v = torch.randn(size=(5, 60, 512))
        output, attn_weights = mha(q, k, v, mask=None)
        self.assertListEqual(list(output.size()), [5, 50, 512])
        self.assertListEqual(list(attn_weights.size()), [5, 8, 50, 60])

    def test_forward_with_mask(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        q = torch.randn(size=(5, 80, 512))
        k = torch.randn(size=(5, 100, 512))
        v = torch.randn(size=(5, 100, 512))
        seq_len = (torch.ones(5, )*3).int()
        padding_mask = helper.create_padding_mask(seq_len, max_len=100)
        output, attn_weights = mha(q, k, v, mask=padding_mask)
        self.assertListEqual(list(output.size()), [5, 80, 512])
        self.assertListEqual(list(attn_weights.size()), [5, 8, 80, 100])
        self.assertEqual(attn_weights[:, :, :, :3].sum(dim=-1).mean(), 1)


if __name__ == '__main__':
    unittest.main()
