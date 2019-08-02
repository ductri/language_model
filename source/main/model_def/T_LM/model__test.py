import unittest

import torch
from torch import nn
from torch.nn import functional as F

from model_def.T_LM.model import Model


class TestModel(unittest.TestCase):

    def test_get_logits(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1, 0, 1)
        seq_len = torch.randint(100, size=(5,))
        x = torch.randint(1000, size=(5, 100))
        output = model.get_logits(x, seq_len)
        self.assertListEqual(list(output.size()), [5, 100, 1000])

    def test_forward(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1, 0, 1)
        model.eval()
        batch_size = 1

        initial_words = torch.randint(1000, size=(batch_size, 1))
        output = model(initial_words, max_length=10)

        self.assertListEqual(list(output.size()), [batch_size, 10])

    def test_forward2(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1, 0, 1)
        model.eval()

        x = torch.tensor([[11, 12, 13, 14]])
        x = F.pad(x, pad=(1, 0), value=model.bos_id)

        output = model.get_logits(x, seq_len=torch.tensor([10]))
        print('out1\n', output.max(dim=-1))

        x = torch.tensor([[11, 12, 13, 16]])
        x = F.pad(x, pad=(1, 0), value=model.bos_id)
        output = model.get_logits(x, seq_len=torch.tensor([10]))
        print('out2\n', output.max(dim=-1))


if __name__ == '__main__':
    unittest.main()

