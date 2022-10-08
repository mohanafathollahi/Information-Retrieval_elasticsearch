"""
.. module:: GroupsComparison

GroupsComparison
******

:Description: GroupsComparison

    Samples documents inside a defined directory, to later perform cosine similarity between them.

:Authors:
    hernandez
    temple

:Version:  0.1

:Date:  24/10/2021
"""
from elasticsearch import Elasticsearch
from itertools import combinations
from os import listdir
import numpy as np
import pandas as pd
import TFIDFViewer as tfidf


def sample_within_groups(path, n_samp):
    groups = sorted(listdir(path))
    rf = [[] for _ in range(3)]
    for i in groups:
        samples = np.random.choice(listdir(path + i), size=n_samp, replace=False)
        for doc1, doc2 in combinations(samples, 2):
            rf[0].append(i)
            rf[1].append(doc1)
            rf[2].append(doc2)

    df = pd.DataFrame({'group': rf[0], 'doc1': rf[1], 'doc2': rf[2]})
    return df


def sample_between_groups(path, n_samp):
    groups = sorted(listdir(path))
    rf = [[] for _ in range(4)]
    counter = 0
    for g1, g2 in combinations(groups, 2):
        np.random.seed(1234 + counter)
        sample1 = np.random.choice(listdir(path + g1), size=n_samp, replace=False)
        sample2 = np.random.choice(listdir(path + g2), size=n_samp, replace=False)
        for i in range(n_samp):
            rf[0].append(g1)
            rf[1].append(sample1[i])
            rf[2].append(g2)
            rf[3].append(sample2[i])
        counter += 1

    df = pd.DataFrame({'group1': rf[0], 'doc1': rf[1], 'group2': rf[2], 'doc2': rf[3]})

    return df


def get_cos_sim(client_, index_, path1_, path2_):
    file1_id = tfidf.search_file_by_path(client_, index_, path1_)
    file2_id = tfidf.search_file_by_path(client_, index_, path2_)

    file1_tw = tfidf.toTFIDF(client_, index_, file1_id)
    file2_tw = tfidf.toTFIDF(client_, index_, file2_id)

    return tfidf.cosine_similarity(file1_tw, file2_tw)


if __name__ == '__main__':

    path_news = '../20_newsgroups/'

    sample_size = 42
    news_within = sample_within_groups(path_news, sample_size)
    news_between = sample_between_groups(path_news, sample_size)
    client = Elasticsearch(timeout=1000)
    index = 'news'

    print(f'Calculating cosine similarity of {index} within groups...')
    nw_similarity = []
    for ix, row in news_within.iterrows():
        path1 = path_news + row.group + '/' + row.doc1
        path2 = path_news + row.group + '/' + row.doc2
        nw_similarity.append(get_cos_sim(client, index, path1, path2))

    news_within['sim'] = nw_similarity
    mean_within = news_within.groupby('group')['sim'].mean().sort_values()
    print(mean_within)
    file_nw = open('news-within.txt', 'w')
    file_nw.write(mean_within.to_string())
    file_nw.close()

    print(f'Calculating cosine similarity of {index} between groups...')
    nb_similarity = []
    for ix, row in news_between.iterrows():
        path1 = path_news + row.group1 + '/' + row.doc1
        path2 = path_news + row.group2 + '/' + row.doc2
        nb_similarity.append(get_cos_sim(client, index, path1, path2))

    news_between['sim'] = nb_similarity
    mean_between = news_between.groupby(['group1', 'group2'])['sim'].mean().sort_values()
    print(mean_between)
    file_nb = open('news-between.txt', 'w')
    file_nb.write(mean_between.to_string())
    file_nb.close()

    print('Done.')
