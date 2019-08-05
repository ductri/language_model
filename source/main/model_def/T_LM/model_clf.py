import torch
from torch import nn, optim
from torch.nn import functional as F

from model_def.transformer import helper


class ModelCLF(nn.Module):
    def __init__(self, model, d_model, num_class):
        super(ModelCLF, self).__init__()
        self.model = model
        self.fc = nn.Linear(d_model, num_class)

    def forward(self, x, seq_len, *args):
        """

        :param x: (..., seq_len, last_hidden_size)
        :param args:
        :return:
        """
        with torch.no_grad():
            logits = self.get_logits(x, seq_len)
            return F.softmax(logits, dim=-1)

    def get_logits(self, x, seq_len):
        """

        :param x: (batch, seq_len)
        :param seq_len: (batch)
        :return:
        """
        def core_get_logits(x, seq_len):
            assert x[0][0] == self.model.bos_id, self.model.bos_id

            x = self.model.word_embedding(x)
            dec_out = x
            look_ahead_mask = helper.create_look_ahead_mask(x.size(1)).to(x.device)
            padding_mask = helper.create_padding_mask(seq_len, x.size(1))
            for i in range(len(self.model.decoder)):
                dec_out, _ = self.model.decoder[i](dec_out, look_ahead_mask, padding_mask)
            return dec_out

        x = F.pad(x, pad=(1, 0), value=self.model.bos_id)
        x = F.pad(x, pad=(0, 1), value=self.model.eos_id)

        # (batch, seq_len, d_model)
        logits = core_get_logits(x, seq_len)

        # (batch, d_model)
        x = logits.mean(dim=1)
        x = self.fc(x)
        return x


class ModelCLFTraining(nn.Module):
    def __init__(self, model_clf):
        super(ModelCLFTraining, self).__init__()
        self.model_clf = model_clf
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.lr = 1e-5
        self.optimizer = optim.Adam(params=self.model_clf.parameters(), lr=self.lr)

    def forward(self, *input):
        pass

    def get_loss(self, x, seq_len, target):
        """

        :param x: (batch, seq_len)
        :param seq_len: (batch)
        :param target: (batch)
        :return:
        """
        logits = self.model_clf.get_logits(x, seq_len)
        loss = self.loss_fn(input=logits, target=target)
        loss = loss.mean(dim=0)
        return loss

    def train_batch(self, x, seq_len, target):
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(x, seq_len, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_lr(self):
        return self.lr
