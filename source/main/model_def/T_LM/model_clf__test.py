import logging
import unittest

import torch
from torch import nn

from model_def.T_LM.model_clf import ModelCLF, ModelCLFTraining
from model_def.T_LM.model import Model


class TestModelCLF(unittest.TestCase):

    def test_get_logits(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1, 0, 1)
        model = ModelCLF(model=model, d_model=512, num_class=2)
        model.eval()
        batch_size = 10

        x = torch.randint(1000, size=(batch_size, 100))
        seq_len = torch.randint(100, size=(batch_size, ))
        logits = model.get_logits(x, seq_len)
        self.assertListEqual(list(logits.size()), [batch_size, 2])

    def test_forward(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1, 0, 1)
        model = ModelCLF(model=model, d_model=512, num_class=2)
        model.eval()
        batch_size = 10

        x = torch.randint(1000, size=(batch_size, 100))
        seq_len = torch.randint(100, size=(batch_size,))
        prob = model(x, seq_len)
        self.assertListEqual(list(prob.size()), [batch_size, 2])
        logging.info(prob[:, 0].cpu().numpy().mean())

    def test_training(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1, 0, 1)
        model = ModelCLF(model=model, d_model=512, num_class=2)
        model = ModelCLFTraining(model)
        model.train()
        batch_size = 10

        x = torch.randint(1000, size=(batch_size, 100))
        seq_len = torch.randint(100, size=(batch_size,))
        target = torch.randint(2, size=(batch_size, ))

        for i in range(50):
            l = model.train_batch(x, seq_len, target)
            if i % 10 == 0:
                logging.info('Step: %s \t Loss: %.3f', i, l)

        model.eval()
        prob = model.model_clf(x, seq_len)
        pred = prob.argmax(dim=-1)
        pred_np = pred.cpu().numpy()
        self.assertEqual((pred_np != target).sum(), 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

