from torch import nn
from torch.nn import functional as F

from model_def.transformer.multi_head_attention import MultiHeadAttention
from model_def.transformer.point_wise_feed_forward import PointWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PointWiseFeedForward(d_model=d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, word_input, mask, *input):
        """

        :param word_input: (batch, seq_len, d_model)
        :param input:
        :return:
        """
        # (batch, seq_len, d_model)
        attn_output, _ = self.mha(word_input, word_input, word_input, mask)
        attn_output = self.dropout1(attn_output)
        out1 = F.layer_norm(word_input + attn_output, normalized_shape=[attn_output.size(-1)])

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = F.layer_norm(out1 + ffn_output, normalized_shape=[out1.size(-1)])

        return out2
