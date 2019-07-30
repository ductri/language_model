import unittest

import torch
from torch import nn

from model_def.T_LM.model import Model
from model_def.T_LM.model_training import ModelTraining


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        word_embedding = nn.Embedding(1000, 512)
        model = Model(word_embedding, 512, 3, 8, 0.1)
        model_training = ModelTraining(model)
        batch = 10
        x = torch.randint(1000, size=(batch, 100))
        seq_len = torch.randint(low=10, high=100, size=(batch,))
        for i in range(100):
            loss = model_training.train_batch(x, seq_len)
            if i % 10 == 0:
                model.eval()
                pred = model(x, seq_len)

                print('Step %s: Loss: %.4f' % (i, loss))
                print('Input: %s' % (x[:3, :10], ))
                print('Predict: %s' % (pred[:3, :10], ))

        model.eval()
        pred = model(x, seq_len)
        self.assertEqual((x[:, :10] != pred[:, :10]).sum(), 0)


if __name__ == '__main__':
    unittest.main()

