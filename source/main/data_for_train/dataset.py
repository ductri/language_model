import logging

import numpy as np
import pandas as pd

import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import torch
import pickle

from model_def.T_LM import constants

voc = None
NUM_WORKERS = 0


class Docs(Dataset):
    def __init__(self, docs):
        super(Docs, self).__init__()

        logging.info('Encoding ...')
        self.docs = [voc.encode_as_ids(doc) for doc in docs]
        content_max_length = constants.MAX_LEN - 2

        # Cut off
        logging.info('Cutting off ...')
        self.docs = [doc[:content_max_length] for doc in self.docs]

        self.length = [len(doc) for doc in self.docs]

        logging.info('Prepend and append ...')
        # Prepend and append bos and eos
        self.docs = [[voc.bos_id()] + doc + [voc.eos_id()] for doc in self.docs]

        logging.info('Padding ...')
        # Padding to adjust same length
        self.docs = [doc + [voc.pad_id()]*(content_max_length - length) for doc, length in zip(self.docs, self.length)]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        word_length = self.length[idx]

        return doc, word_length


def bootstrap():
    global voc
    voc = spm.SentencePieceProcessor()
    voc.load(constants.voc_path)

    logging.info('Vocab from file %s contains %s tokens', constants.voc_path, voc.get_piece_size())


def collate_fn(list_data):
    """
    shape == (batch_size, col1, col2, ...)
    """
    data = zip(*list_data)
    data = [np.stack(col, axis=0) for col in data]
    data = [torch.from_numpy(col) for col in data]
    return data


def get_ds_from_csv(path):
    df = pd.read_csv(path, lineterminator='\n', nrows=100)
    docs = list(df['text'])
    docs = Docs(docs)
    logging.info('Data at %s has size: %s', path, len(docs))
    return docs


def get_datasets(batch_size=64):
    docs = get_ds_from_csv(constants.train_path)
    train_loader = DataLoader(docs, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    logging.info('TRAIN: There will be %s step/epoch', len(train_loader))

    docs = get_ds_from_csv(constants.eval_path)
    logging.info('Data eval size: %s', len(docs))
    eval_loader = DataLoader(docs, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    logging.info('TEST: There will be %s step/epoch', len(eval_loader))

    return train_loader, eval_loader


def dump_dataset(ds, path_to_file):
    with open(path_to_file, 'wb') as o_f:
        pickle.dump(ds, o_f)


def get_datasets_2(batch_size=64):
    docs = pickle.load(open('/source/main/data_for_train/output/v2_social_and_wiki_train.pkl', 'rb'))
    train_loader = DataLoader(docs, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    logging.info('TRAIN: There will be %s step/epoch', len(train_loader))

    docs = pickle.load(open('/source/main/data_for_train/output/v2_social_and_wiki_eval.pkl', 'rb'))
    logging.info('Data eval size: %s', len(docs))
    eval_loader = DataLoader(docs, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    logging.info('TEST: There will be %s step/epoch', len(eval_loader))

    return train_loader, eval_loader
