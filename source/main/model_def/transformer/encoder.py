from torch import nn
import torch

from model_def.transformer.encoder_layer import EncoderLayer
from model_def.transformer.position_embedding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, word_embedding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.word_embedding = word_embedding
        self.pos_embedding = PositionalEncoding(max_seq_len=100, d_model=d_model)
        self.enc_layers = nn.ModuleList(EncoderLayer(d_model, num_heads, rate) for _ in range(num_layers))
        self.dropout = nn.Dropout(rate)

    def forward(self, word_input, padding_mask, *input):
        """

        :param word_input: (batch, max_seq_len)
        :param padding_mask: Provide info about length
        :param input:
        :return:
        """
        x = self.word_embedding(word_input)
        x *= torch.sqrt(self.d_model)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, padding_mask)

        # (batch, seq_len, d_model)
        return x
