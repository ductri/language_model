import torch
from torch import nn
from torch.nn import functional as F

from model_def.transformer.multi_head_attention import MultiHeadAttention
from model_def.transformer.point_wise_feed_forward import PointWiseFeedForward


class DecoderLayerCustomized(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(DecoderLayerCustomized, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PointWiseFeedForward(d_model=d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, word_input, look_ahead_mask, target_padding_mask, *args):
        """

        :param word_input: (batch, target_seq_len, d_model)
        :param look_ahead_mask: (batch, target_seq_len, )
        :param target_padding_mask: (batch, target_seq_len, )
        :param args:
        :return:
        """

        # (batch, target_seq_len, d_model)
        if target_padding_mask is None and look_ahead_mask is None:
            combined_mask = None
        elif target_padding_mask is not None and look_ahead_mask is not None:
            combined_mask = torch.max(target_padding_mask, look_ahead_mask)
        else:
            combined_mask = target_padding_mask if target_padding_mask is not None else look_ahead_mask

        attn1, attn_weights_block1 = self.mha(word_input, word_input, word_input, mask=combined_mask)
        attn1 = self.dropout1(attn1)
        out1 = F.layer_norm(word_input + attn1, normalized_shape=[attn1.size(-1)])

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        output = F.layer_norm(out1 + ffn_output, normalized_shape=[ffn_output.size(-1)])

        return output, attn_weights_block1
