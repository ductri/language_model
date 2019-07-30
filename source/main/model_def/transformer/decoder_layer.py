import torch
from torch import nn
from torch.nn import functional as F

from model_def.transformer.multi_head_attention import MultiHeadAttention
from model_def.transformer.point_wise_feed_forward import PointWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PointWiseFeedForward(d_model=d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, word_input, encoder_output, look_ahead_mask, source_padding_mask, target_padding_mask, *args):
        """

        :param encoder_output: (batch, source_seq_len, d_model)
        :param word_input: (batch, target_seq_len, d_model)
        :param look_ahead_mask: (batch, target_seq_len, )
        :param source_padding_mask: (batch, source_seq_len, )
        :param target_padding_mask: (batch, target_seq_len, )
        :param args:
        :return:
        """

        # (batch, target_seq_len, d_model)
        combined_mask = torch.max(target_padding_mask, look_ahead_mask)
        attn1, attn_weights_block1 = self.mha1(word_input, word_input, word_input, mask=combined_mask)
        attn1 = self.dropout1(attn1)
        out1 = F.layer_norm(word_input + attn1, normalized_shape=[attn1.size(-1)])

        # (batch, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(q=out1, k=encoder_output, v=encoder_output, mask=source_padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = F.layer_norm(out1 + attn2, normalized_shape=[attn2.size(-1)])

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        output = F.layer_norm(out2 + ffn_output, normalized_shape=[ffn_output.size(-1)])

        return output, attn_weights_block1, attn_weights_block2
