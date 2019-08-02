import torch
from torch import nn, optim
from torch.nn import functional as F
from naruto_skills import pytorch_utils


class ModelTraining(nn.Module):
    def __init__(self, model):
        super(ModelTraining, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.lr = 4e-5
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

    def forward(self, *args):
        """

        :param args:
        :return:
        """
        pass

    def get_loss(self, x, seq_len):
        """

        :param x: (batch, seq_len)
        :param seq_len: (batch)
        :return:
        """
        # source = F.pad(x, pad=(1, 0), value=self.model.bos_id)
        # target = F.pad(x, pad=(0, 1), value=self.model.eos_id)
        assert x[0][0] == self.model.bos_id, self.model.bos_id

        source = x[:, :-1]
        target = x[:, 1:]
        seq_len = seq_len + 1

        # (batch, seq_len, vocab_size)
        logits = self.model.get_logits(source, seq_len)
        # (batch, vocab_size, seq_len)
        logits = logits.transpose(2, 1)

        # (batch, seq_len)
        loss = self.loss_fn(logits, target)
        max_len = source.size(1)
        loss_mask = pytorch_utils.length_to_mask(seq_len, max_len=max_len, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)
        loss = torch.div(loss.sum(dim=1), seq_len.float())
        loss = loss.mean(dim=0)
        return loss

    def train_batch(self, x, seq_len):
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(x, seq_len)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_lr(self):
        return self.lr
