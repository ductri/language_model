import unittest
import time

import torch
from torch import nn
from naruto_skills import pytorch_utils

from model_def.T_LM.model import Model
from model_def.T_LM.model_training import ModelTraining


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word_embedding = nn.Embedding(1000, 512).to(device)
        model = Model(word_embedding, 512, 3, 8, 0.1).to(device)
        model_training = ModelTraining(model).to(device)
        batch = 10
        x = torch.randint(1000, size=(batch, 100)).to(device)
        seq_len = torch.randint(low=10, high=100, size=(batch,)).to(device)
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
        self.assertEqual((x[:, 1:10] != pred[:, :9]).sum(), 0)

    def test_speed(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word_embedding = nn.Embedding(30000, 512).to(device)
        model = Model(word_embedding, 512, 6, 8, 0.1).to(device)
        model_training = ModelTraining(model).to(device)
        print('Total params: %s' % pytorch_utils.count_parameters(model_training))
        batch = 64
        max_len = 100
        x = torch.randint(1000, size=(batch, max_len)).to(device)
        seq_len = torch.randint(low=30, high=100, size=(batch,)).to(device)
        for i in range(100):
            start = time.time()
            loss = model_training.train_batch(x, seq_len)
            end = time.time()
            if i % 10 == 0:
                model.eval()
                model(x, seq_len)
                print('Step: %s Loss: %.4f Duration: %.4f s' % (i, loss, end-start))


if __name__ == '__main__':
    unittest.main()

