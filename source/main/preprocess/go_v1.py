"""
This file is supposed to be run once
"""
from pathlib import Path
import ast

import pandas as pd


if __name__ == '__main__':
    root_dir = '/source/main/data_download/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/'
    root_dir = Path(root_dir)
    num_rows = 0
    print('aaa')
    all_files = list(root_dir.glob('*.csv'))

    for idx, topic_file in enumerate(all_files):
        try:
            print('%s/%s - %s' % (idx, len(all_files), topic_file.name))
            df = pd.read_csv(str(topic_file), usecols=['id', 'topic_id', 'search_text', 'mention_type'])

            df = df.dropna(subset=['search_text'])
            df = df[df['search_text'].map(lambda x: len(x.split()) > 1)]
            df['mention'] = df['search_text'].map(lambda x: ast.literal_eval(x)[1])
            df = df[df['mention'].map(lambda x: len(x) > 0)]
            df.drop_duplicates(subset=['mention'])
            num_rows += df.shape[0]
            df[['id', 'topic_id', 'mention', 'search_text']].to_csv('/source/main/data_for_train/output/topics_v2_2018-01-01T00:00:0_2019-01-01T00:00:00/v1/%s' % topic_file.name, index=None)
        except Exception as e:
            print('Exception at topic: %s' % topic_file)
    print('Total rows: %s' % num_rows)
