# -*- coding: utf-8 -*-
"""Utilities for preprocessing sequence data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import random
import itertools


def make_sampling_table(size, sampling_factor=1e-5):
    """Generates a word rank-based probabilistic sampling table.

    Used for generating the `sampling_table` argument for `skipgrams`.
    `sampling_table[i]` is the probability of sampling
    the word i-th most common word in a dataset
    (more common words should be sampled less frequently, for balance).

    The sampling probabilities are generated according
    to the sampling distribution used in word2vec:

    `p(word) = min(1, sqrt(word_frequency / sampling_factor) / (word_frequency / sampling_factor))`

    We assume that the word frequencies follow Zipf's law (s=1) to derive
    a numerical approximation of frequency(rank):

    `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
    where `gamma` is the Euler-Mascheroni constant.

    # Arguments
        size: Int, number of possible words to sample.
        sampling_factor: The sampling factor in the word2vec formula.

    # Returns
        A 1D Numpy array of length `size` where the ith entry
        is the probability that a word of rank i should be sampled.
    """
    gamma = 0.577
    rank = np.arange(size)
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
    f = sampling_factor * inv_fq

    return np.minimum(1., f / np.sqrt(f))


def neg_sample_remix(target_index, context, w2i, max_num=10000):
    neg_words = []
    left_str = ''
    right_str = ''

    for i in range(target_index):
        if context[i] != 'BOS':
            left_str += context[i]
    for i in range(min(target_index + 1, len(context) - 1), len(context)):
        if context[i] != 'EOS':
            right_str += context[i]

    for seq in [left_str, right_str]:
        if len(seq) > 1:
            sub_str = [seq[i:i + x + 1] for x in range(len(seq)) for i in range(len(seq) - x) if
                       seq[i:i + x + 1] in w2i and seq[i:i + x + 1] not in context]
            neg_words += sub_str

    neg_words = list(set(neg_words))

    if max_num < len(neg_words):
        neg_words = random.sample(neg_words, max_num)

    return neg_words


def trans_w2i(text, w2i):  # text = list of words
    index = []
    for i in text:
        if i in w2i:
            index.append(w2i[i])
        else:
            index.append(0)
    return index


def skipgram_fix(text, vocabulary_size, w2i, w2c, max_neg=10000,
                 window_size=4, negative_samples=1, shuffle=True,
                 categorical=False, sampling_table='zipf', neg_sampling_table='uniform', sampling_fix=True,
                 neg_self=True, seed=None):
    couples_all = []
    labels_all = []
    pos_num = 0

    i2w = dict(zip(w2i.values(), w2i.keys()))
    i2c = {}
    for w in w2c:
        i2c[w2i[w]] = w2c[w]

    t = 1.0 * 1e-5
    if sampling_table == 'freq':
        sampling_table = {}
        sum_count = sum(i2c.values())
        for i, c in i2c.items():
            p = np.sqrt(t * sum_count / c)
            sampling_table[i] = p
    elif sampling_table == 'zipf':
        samp_table = make_sampling_table(vocabulary_size, sampling_factor=t)
        sampling_table = {}
        for e, p in enumerate(samp_table, start=1):
            sampling_table[e] = p

    if neg_sampling_table == 'unigram':
        neg_sampling_temp = {}
        i2c_neg = {}
        for i, c in i2c.items():
            i2c_neg[i] = np.power(c, 3 / 4)
        sum_power = sum(i2c_neg.values())
        for i, c in i2c_neg.items():
            neg_sampling_temp[i] = c / sum_power
        key = np.array(list(neg_sampling_temp.keys()))
        value = np.array(list(neg_sampling_temp.values()))
        bins = np.cumsum(value)

    for sent in text:  # text = [[a,b,c], [d,e,f]]
        couples = []
        labels = []
        couples_neg_fix = []
        labels_neg_fix = []
        couples_neg_self = []
        labels_neg_self = []
        sequence = trans_w2i(sent, w2i)  # seq = [1,2,3]

        for i, wi in enumerate(sequence):
            if not wi:
                continue
            if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    if not sampling_fix:
                        continue
                    else:
                        word = i2w[wi]
                        if len(word) == 1:
                            continue
                        else:
                            sub_str = [word[j:j + x + 1] for x in range(len(word) - 1) for j in range(len(word) - x)]
                            sub_score = [sampling_table[w2i[sub]] for sub in sub_str if sub in w2i]
                            max_score = max(sub_score) if len(sub_score) > 1 else 1.0
                            if max_score > sampling_table[wi] / 2.0:
                                continue

            window_start = max(0, i - window_size)
            window_end = min(len(sequence), i + window_size + 1)

            for j in range(window_start, window_end):
                if j != i:
                    wj = sequence[j]
                    if not wj:
                        continue
                    couples.append(sorted([wi, wj]))
                    if categorical:
                        labels.append([0, 1])
                    else:
                        labels.append(1)
                    pos_num += 1

                    if negative_samples > 0:
                        num = 0
                        while num < negative_samples:
                            if neg_sampling_table == 'unigram':
                                w_neg = key[np.digitize(np.random.random_sample(1), bins)][0]
                            elif neg_sampling_table == 'uniform':
                                w_neg = random.randint(1, vocabulary_size - 1)
                            couples.append(sorted([wi, w_neg]))
                            if categorical:
                                labels.append([1, 0])
                            else:
                                labels.append(0)
                            num += 1

            if max_neg > 0:
                context = sent[window_start: window_end]
                target_index = i - window_start
                neg_fix_word = neg_sample_remix(target_index, context, w2i, max_neg)
                neg_fix_index = trans_w2i(neg_fix_word, w2i)
                couples_neg_fix += [sorted([wi, wn]) for wn in neg_fix_index]
                if categorical:
                    labels_neg_fix += [[1, 0]] * len(neg_fix_index)
                else:
                    labels_neg_fix += [0] * len(neg_fix_index)

            if neg_self:
                tar_word = i2w[wi]
                if len(tar_word) > 1:
                    sub_str = list(set([tar_word[i:i + x + 1] for x in range(len(tar_word) - 1)
                                        for i in range(len(tar_word) - x) if tar_word[i:i + x + 1] in w2i]))
                    if len(sub_str) > 1:
                        neg_self_num = 0
                        for t in list(itertools.combinations(sub_str, 2)):
                            neg_self_cp = trans_w2i(t, w2i)
                            couples_neg_self.append(neg_self_cp)
                            neg_self_num += 1
                        if categorical:
                            labels_neg_self += [[1, 0]] * neg_self_num
                        else:
                            labels_neg_self += [0] * neg_self_num

        couples_all += couples + couples_neg_fix + couples_neg_self
        labels_all += labels + labels_neg_fix + labels_neg_self

    neg_num = len(labels_all) - pos_num

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples_all)
        random.seed(seed)
        random.shuffle(labels_all)

    word_np = np.empty(shape=len(labels_all), dtype=np.int)
    context_np = np.empty(shape=len(labels_all), dtype=np.int)
    label_np = np.empty(shape=len(labels_all), dtype=np.int)

    for i in range(len(labels_all)):
        word_np[i] = couples_all[i][0]
        context_np[i] = couples_all[i][1]
        label_np[i] = labels_all[i]

    # print(pos_num, neg_num)

    return (word_np, context_np), label_np, pos_num, neg_num
