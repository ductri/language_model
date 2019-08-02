"""
This file is supposed to be run once
"""
import logging
import ast
import time

import pandas as pd
import dask.dataframe as dd

from preprocess import preprocessor


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start = time.time()
    root_dir = '/source/main/data_download/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/'
    num_rows = 0
    df = dd.read_csv(root_dir + '/*.csv', usecols=['id', 'topic_id', 'search_text'], blocksize='64Mb')
    logging.info('Dask is processing with %s partitions', df.npartitions)
    # df = df.repartition(npartitions=8)
    # logging.info('Dask is processing with %s partitions', df.npartitions)
    logging.info('Drop NA and duplicates ... ')
    df = df.dropna(subset=['search_text'])
    logging.info('Fining search_text ...')
    df = df[df['search_text'].map(lambda x: len(x) > 1)]
    df['mention'] = df['search_text'].map(lambda x: preprocessor.train_preprocess(ast.literal_eval(x)[1], max_length=100), meta=pd.Series([], dtype=str, name='mention'))
    df = df[df['mention'].map(lambda x: len(x) > 1)]
    df.drop_duplicates(subset=['mention'])
    logging.info('Saving ...')
    #df[['id', 'topic_id', 'mention']].to_csv('/source/main/data_for_train/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/*.csv')
    logging.info('Mean len: %s', df['mention'].map(len).mean().compute())
    logging.info('Duration: %.2f s', time.time() - start)
