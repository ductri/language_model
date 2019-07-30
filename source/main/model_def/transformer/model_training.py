from torch import nn

from model_def.transformer.encoder import Encoder
from model_def.transformer.decoder import Decoder

from model_def.transformer import helper


class ModelTraining(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, encoder_embedding, decoder_embedding, dropout_rate=0.1):
        super(ModelTraining, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, encoder_embedding, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, decoder_embedding, dropout_rate)
        self.final_layer = nn.Linear(d_model, decoder_embedding.weight.size(0))

    def forward(self, source_words, target_words, source_length, target_length, *args):
        """

        :param source_words: (batch, max_seq_len_1)
        :param target_words: (batch, max_seq_len_2)
        :param source_length:  (batch)
        :param target_length:  (batch)
        :param args:
        :return: logits (batch, max_seq_len_2, target_vocab_size)
        """
        source_padding = helper.create_padding_mask(source_seq_len=source_length, max_len=source_words.size(-1))
        encoding_output = self.encoder(source_words, source_padding)
        look_ahead_mask = helper.create_look_ahead_mask(target_words.size(-1))
        decoding_output = self.decoder(target_words, encoding_output, look_ahead_mask, source_padding)
        output = self.final_layer(decoding_output)
        return output

