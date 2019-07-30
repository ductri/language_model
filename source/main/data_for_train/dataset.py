import logging

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch
from naruto_skills.new_voc import Voc

from model_def.transformer import constants


voc = None
NUM_WORKERS = 0


class Docs(Dataset):
    def __init__(self, docs):
        super(Docs, self).__init__()
        # ASSERT __start__ and __end__
        assert docs[0].split()[0] == '__s__'
        assert docs[0].split()[-1] == '__e__'

        self.docs = voc.docs2idx(docs, equal_length=constants.MAX_LEN)
        self.length = [len(doc.split()) for doc in docs]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        word_length = self.length[idx]

        return doc, word_length


def bootstrap():
    global voc
    path_src = constants.voc_path
    voc = Voc.load(path_src)

    logging.info('Vocab from file %s contains %s tokens', path_src, len(voc.index2word))


def collate_fn(list_data):
    """
    shape == (batch_size, col1, col2, ...)
    """
    data = zip(*list_data)
    data = [np.stack(col, axis=0) for col in data]
    data = [torch.from_numpy(col) for col in data]
    return data


def get_datasets(batch_size=64):
    train_path = '/source/main/data_for_train/output/train.csv'
    eval_path = '/source/main/data_for_train/output/eval.csv'

    df = pd.read_csv(train_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['text'] = df['text'].map(lambda x: ' '.join(x.split()[:constants.MAX_LEN]))
    docs = list(df['text'])
    docs = Docs(docs)
    train_loader = DataLoader(docs, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    logging.info('TRAIN: There will be %s step/epoch', len(train_loader))

    df = pd.read_csv(eval_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['text'] = df['text'].map(lambda x: ' '.join(x.split()[:constants.MAX_LEN]))
    docs = list(df['text'])
    docs = Docs(docs)
    eval_loader = DataLoader(docs, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    logging.info('TEST: There will be %s step/epoch', len(eval_loader))

    return train_loader, eval_loader

