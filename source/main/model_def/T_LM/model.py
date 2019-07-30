import torch
from torch import nn

from model_def.T_LM.decoder_layer_customized import DecoderLayerCustomized
from model_def.T_LM.layer_lm import LayerLM
from model_def.transformer import helper


class Model(nn.Module):
    def __init__(self, word_embedding, d_model, num_layers, num_heads, rate):
        super(Model, self).__init__()
        self.word_embedding = word_embedding
        self.decoder = nn.ModuleList([DecoderLayerCustomized(d_model, num_heads, rate) for _ in range(num_layers)])
        self.lm_layer = LayerLM(last_hidden_size=d_model, vocab_size=self.word_embedding.num_embeddings)

    def forward(self, x, seq_len, *args):
        """

        :param x: (batch, seq_len)
        :param seq_len: (batch)
        :param args:
        :return:
        """
        logits = self.get_logits(x, seq_len)
        return torch.argmax(logits, dim=-1)

    def get_logits(self, x, seq_len):
        """

        :param x: (batch, seq_len)
        :param seq_len: (batch)
        :param args:
        :return:
        """
        x = self.word_embedding(x)
        dec_out = x
        look_ahead_mask = helper.create_look_ahead_mask(x.size(1))
        padding_mask = helper.create_padding_mask(seq_len, x.size(1))
        for i in range(len(self.decoder)):
            dec_out, _ = self.decoder[i](dec_out, look_ahead_mask, padding_mask)

        logits = self.lm_layer.get_logits(dec_out)
        return logits

