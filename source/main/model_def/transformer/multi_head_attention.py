import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_head(self, x):
        """

        :param x: (batch, seq_len, d_model)
        :return:
        """
        x = x.view(x.size(0), x.size(1), self.num_heads, self.depth)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, q, k, v, mask, *input):
        """

        :param q: (batch, seq_len, d_model)
        :param k:
        :param v:
        :param mask:
        :param input:
        :return: (batch, seq_len_q, d_model)
        """

        # (batch, seq_len_q, depth_q)
        q = self.wq(q)

        # (batch, seq_len_k, depth_k)
        k = self.wk(k)

        # (batch, seq_len_v, depth_v)
        v = self.wv(v)

        # (batch, num_head, seq_len_*, depth_*)
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        # attn: (batch, num_head, seq_len_q, depth_v)
        attn, attn_weights = MultiHeadAttention.scaled_dot_product_attention(q, k, v, mask=mask)
        # (batch, seq_len_q, num_head, depth_v)
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.contiguous()
        # (batch, seq_len_q, d_model)
        attn = attn.view(attn.size(0), attn.size(1), self.d_model)

        output = self.dense(attn)
        return output, attn_weights

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        """

        :param q: (..., seq_len_q, depth)
        :param k: (..., seq_len_k, depth)
        :param v: (..., seq_len_v, depth_v)
        :param mask: (..., seq_len_q, seq_len_k). 1 is masking, 0 is normal
        :return: output, att_weights
        - output: (..., seq_len_q, depth_v)
        - attn_weights: (..., seq_len_q, seq_len_k)
        """
        assert k.size(1) == v.size(1)

        # (..., seq_len_q, seq_len_k)
        attn = q.matmul(k.transpose(-1, -2))
        attn = attn / np.sqrt(q.size(-1))

        # add the mask to the scaled tensor.
        if mask is not None:
            attn += (mask * -1e9)

        # (..., seq_len_q, seq_len_k)
        attn_weights = F.softmax(attn, dim=-1)

        # (..., seq_len_q, seq_len_k, 1)
        # *
        # (..., 1        , seq_len_v, depth_v)
        output = torch.mul(attn_weights[..., :, :, None], v[..., None, :, :])

        # (..., seq_len_q, depth_v)
        output = output.sum(dim=-2)

        return output, attn_weights
