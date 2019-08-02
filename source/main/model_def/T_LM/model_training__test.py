import unittest
import time

import torch
from torch import nn
from naruto_skills import pytorch_utils

from model_def.T_LM.model import Model
from model_def.T_LM.model_training import ModelTraining
from model_def.T_LM import constants


class TestEncoder(unittest.TestCase):

    def test_forward(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word_embedding = nn.Embedding(1000, 512).to(device)
        model = Model(word_embedding, 512, 3, 8, 0.1, 998, 999).to(device)
        model_training = ModelTraining(model).to(device)
        batch = 5
        max_len = 10
        x = torch.randint(900, size=(batch, max_len)).to(device)
        seq_len = torch.ones(batch).int().to(device) * max_len

        for i in range(200):
            loss = model_training.train_batch(x, seq_len)
            if i % 10 == 0:
                model.eval()
                print('Step %s: Loss: %.4f' % (i, loss))

        model.eval()
        pred = model(x[:, :3], max_len)
        self.assertEqual((x != pred).sum(), 0)

    @unittest.skip('')
    def test_speed(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word_embedding = nn.Embedding(30000, constants.d_model).to(device)
        model = Model(word_embedding, d_model=constants.d_model, num_layers=constants.num_layers,
                      num_heads=constants.num_heads, rate=constants.rate, bos_id=998, eos_id=999).to(device)
        model_training = ModelTraining(model).to(device)
        print('Total params: %s' % pytorch_utils.count_parameters(model_training))

        batch = 16
        x = torch.randint(1000, size=(batch, constants.MAX_LEN)).to(device)
        seq_len = torch.randint(low=30, high=100, size=(batch,)).to(device)
        for i in range(100):
            start = time.time()
            loss = model_training.train_batch(x, seq_len)
            end = time.time()
            if i % 10 == 0:
                model.eval()
                model(x, constants.MAX_LEN)
                print('Step: %s Loss: %.4f Duration: %.4f s' % (i, loss, end - start))


if __name__ == '__main__':
    unittest.main()

