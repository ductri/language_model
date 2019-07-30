from torch import nn

from model_def.transformer.encoder import Encoder
from model_def.transformer.decoder import Decoder

from model_def.transformer import helper


class ModelInfer(nn.Module):
    def __init__(self, model_training, start_tok_idx, end_tok_idx):
        super(ModelInfer, self).__init__()
        self.model_training = model_training
        self.start_tok_idx = start_tok_idx
        self.end_tok_idx = end_tok_idx

    def forward(self, source_words, source_length, max_length, *args):
        """

        :param source_words: (batch, max_seq_len_1)
        :param source_length:  (batch)
        :param max_length: scala int
        :param args:
        :return: predict_output (batch, max_seq_len_2)
        """
        source_padding = helper.create_padding_mask(source_seq_len=source_length, max_len=source_words.size(-1))
        encoding_output = self.encoder(source_words, source_padding)
        for i in range(max_length):
            decoding_output = self.model_training(source_words, target_words, encoding_output, look_ahead_mask, source_padding)
        output = self.final_layer(decoding_output)
        return output

