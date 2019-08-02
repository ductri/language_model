import torch
from torch import nn
from torch.nn import functional as F

from model_def.T_LM.decoder_layer_customized import DecoderLayerCustomized
from model_def.T_LM.layer_lm import LayerLM
from model_def.transformer import helper


class Model(nn.Module):
    def __init__(self, word_embedding, d_model, num_layers, num_heads, rate, bos_id, eos_id):
        super(Model, self).__init__()
        self.word_embedding = word_embedding
        self.decoder = nn.ModuleList([DecoderLayerCustomized(d_model, num_heads, rate) for _ in range(num_layers)])
        self.lm_layer = LayerLM(last_hidden_size=d_model, vocab_size=self.word_embedding.num_embeddings)
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(self, initial_words, max_length,  *args):
        """
        Used for prediction
        :param initial_words: (batch, initial_length)
        :param max_length: scala int
        :param args:
        :return:
        """
        current_tok = torch.tensor([-100])
        batch_size = initial_words.size(0)
        seq_len = torch.ones(batch_size).int().to(initial_words.device) * max_length

        # (batch, initial_length + 1)
        input_words = F.pad(initial_words, pad=(1, 0), value=self.bos_id)
        input_words = F.pad(input_words, pad=(0, max_length-input_words.size(-1)+1), value=0)
        current_length = initial_words.size(-1)

        while current_length+1 <= max_length and (current_tok != self.eos_id).sum().cpu() != 0:
            logits = self.get_logits(input_words, seq_len=seq_len)
            current_tok = torch.argmax(logits[:, current_length], dim=-1)
            input_words[:, current_length+1] = current_tok
            current_length += 1
        return input_words[:, 1:]

    def get_logits(self, x, seq_len):
        """

        :param x: (batch, seq_len)
        :param seq_len: (batch)
        :param args:
        :return:
        """
        x = self.word_embedding(x)
        dec_out = x
        look_ahead_mask = helper.create_look_ahead_mask(x.size(1)).to(x.device)
        padding_mask = helper.create_padding_mask(seq_len, x.size(1))
        for i in range(len(self.decoder)):
            dec_out, _ = self.decoder[i](dec_out, look_ahead_mask, padding_mask)

        logits = self.lm_layer.get_logits(dec_out)
        return logits

    @staticmethod
    def get_word_embedding(weights=None):
        if weights is None:
            return nn.Embedding(30000, embedding_dim=640)
        else:
            return nn.Embedding.from_pretrained(torch.from_numpy(weights).float())
