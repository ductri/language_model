"""
This file is supposed to be run once
"""
import logging
from pathlib import Path
import ast

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from preprocess import preprocessor


def preprocess_file(topic_file):
    df = pd.read_csv(str(topic_file), usecols=['id', 'topic_id', 'search_text'])

    df = df.dropna(subset=['search_text'])
    logging.info('Fining search_text ...')
    df = df[df['search_text'].map(lambda x: len(x.split()) > 1)]
    df['mention'] = df['search_text'].map(
        lambda x: preprocessor.train_preprocess(ast.literal_eval(x)[1], max_length=100))
    df = df[df['mention'].map(lambda x: len(x) > 0)]
    df.drop_duplicates(subset=['mention'])
    df[['id', 'topic_id', 'mention']].to_csv(
        '/source/main/data_for_train/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/v2/%s' % topic_file.name, index=None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    root_dir = '/source/main/data_download/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/'
    root_dir = Path(root_dir)
    Parallel(n_jobs=4)(delayed(preprocess_file)(topic_file) for topic_file in tqdm(list(root_dir.glob('*.csv'))))
