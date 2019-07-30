from torch import nn
import torch
import numpy as np

from model_def.transformer.decoder_layer import DecoderLayer
from model_def.transformer.position_embedding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, word_embedding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.word_embedding = word_embedding
        self.pos_embedding = PositionalEncoding(max_seq_len=100, d_model=d_model)
        self.decoder_layers = nn.ModuleList(DecoderLayer(d_model, num_heads, rate) for _ in range(num_layers))
        self.dropout = nn.Dropout(rate)

    def forward(self, word_input, encoder_output, look_ahead_mask, source_padding_mask, target_padding_mask, *input):
        """

        :param word_input: (batch, max_seq_len)
        :param encoder_output: (batch, target_seq_len, d_model)
        :param source_padding_mask: Provide info about length
        :param target_padding_mask: Provide info about length
        :param input:
        :return:
        """
        x = self.word_embedding(word_input)
        x *= np.sqrt(self.d_model)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.decoder_layers[i](x, encoder_output, look_ahead_mask, source_padding_mask, target_padding_mask)

        # (batch, target_seq_len, d_model)
        return x
