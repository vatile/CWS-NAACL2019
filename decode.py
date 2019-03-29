from collections import namedtuple
from copy import deepcopy
import re
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import numpy as np


class Hypo:

    def __init__(self, prob=0, t=0, seg={'SEG': [['BOS']], 'BUF': []}, m=1):
        self.prob = prob
        self.t = t
        self.seg = seg
        self.m = m
        self.hypo = namedtuple("hypo", "prob, t, seg, m")

    def get_hypo(self):
        return self.hypo(prob=self.prob, t=self.t, seg=self.seg, m=self.m)


def next_prob(h, h_next, window, sim, w2v):
    h_next_words = [''.join(w) for w in h_next.seg['SEG']]  # ['abc', 'de', 'f']
    if h.m == h_next.m:
        return h.prob
    else:
        prob_i = []
        for i in range(1, h_next.m - h.m + 1):
            prob_j = []
            center = h.m - 1 + i
            for j in range(1, min(window + 1, center + 1)):
                # print(center, j, h_next_words[center], h_next_words[center - j])
                try:
                    prob_j.append(sim['|'.join(sorted([h_next_words[center], h_next_words[center - j]]))])
                except Exception:
                    prob_j.append(float(cos_sim(np.asarray(w2v[h_next_words[center]], dtype='float32').reshape(1, -1),
                                                np.asarray(w2v[h_next_words[center - j]], dtype='float32').reshape(1,
                                                                                                                   -1))))
            prob_j = sum(prob_j) / len(prob_j)
            prob_i.append(prob_j)
        return ((h.m - 1) * h.prob + sum(prob_i)) / (h_next.m - 1)


def next_hypo(h, char, seq_len, window, max_word, sim, w2v):
    # h_1: append, h_2: new semi-word (or new complete word)
    h_1 = deepcopy(h)
    h_2 = deepcopy(h)
    h_next = []

    if h.t == 0:
        h_1 = None
        # make char the beginning of a new semi-word
        h_2.seg['BUF'] = [char]
    elif h.t == seq_len - 2:
        h_1 = None
        if ''.join(h_2.seg['BUF']) in w2v:
            # change the last semi-word to a complete word, and make char(['EOS']) a new complete word
            h_2.seg['SEG'].append(h_2.seg['BUF'])
            h_2.seg['SEG'].append([char])
            h_2.seg['BUF'] = []
            h_2.m += 2
        else:
            h_2 = None
    else:
        if len(h_1.seg['BUF']) < max_word:
            # append char to the last semi-word
            h_1.seg['BUF'].append(char)
        else:
            h_1 = None
        if ''.join(h_2.seg['BUF']) in w2v:
            # change the last semi-word to a complete word, and make char the beginning of a new semi-word
            h_2.seg['SEG'].append(h_2.seg['BUF'])
            h_2.seg['BUF'] = [char]
            h_2.m += 1
        else:
            h_2 = None

    for i in [h_1, h_2]:
        if i:
            i.t += 1
            i.prob = next_prob(h, i, window, sim, w2v)
            h_next.append(i)

    return h_next


def decoder(seq, window, max_word, sim, w2v, beam):
    seq_len = len(seq)

    h_all = [[Hypo()]]
    for i in range(seq_len - 1):
        h_all.append([])

    def decode_beam(beam_true):
        for i, char in enumerate(seq[1:], start=1):
            for h in h_all[i - 1]:
                new_hs = next_hypo(h, char, seq_len, window, max_word, sim, w2v)
                for new_h in new_hs:
                    h_all[i].append(new_h)
            h_all[i] = sorted(h_all[i], key=lambda x: x.prob, reverse=True)[:min(len(h_all[i]), beam_true)]
            if h_all[i] == []:
                return max([len(h_all[k]) for k in range(1, i)])
        return h_all

    beam_true = beam
    while True:
        h_all = decode_beam(beam_true)
        if isinstance(h_all, list):
            break
        else:
            if h_all < beam_true:
                # print(h_all)
                print("Something went wrong; couldn't find any possible segmentation.")
                h_all = [[Hypo(prob=0, t=0, seg={'SEG': ['BOS', 'EOS']}, m=0)]]
                return ''.join(seq[1:-1]), 0
            beam_true += 10
            print("Beam size increased to {}".format(beam_true))
            h_all = [[Hypo()]]
            for i in range(seq_len - 1):
                h_all.append([])

    best = h_all[-1][0].seg['SEG'][1:-1]  # best = [[a,b], [c,d]]
    score = h_all[-1][0].prob

    final_seg = ' '.join([''.join(i) for i in best])

    return final_seg, score  # 'a bc', 0.25


def decode_split_by_punc(sent, window, max_word, sim, w2v, beam):  # sent = 'abc'
    pat = r'[0-9０-９a-zA-Z\u4e00-\u9fa5]'
    seq_all = []
    score_all = []
    seq = ['BOS']
    for e, c in enumerate(sent, start=1):
        if re.match(pat, c):
            seq.append(c)  # seq = [a, b]
            if e == len(sent):
                seq.append('EOS')
                par_seg, score = decoder(seq, window, max_word, sim, w2v, beam)
                seq_all.append(par_seg)
                score_all.append(score)
                seq = ['BOS']
        else:
            if len(seq) != 1:
                seq.append('EOS')
                par_seg, score = decoder(seq, window, max_word, sim, w2v, beam)
                seq_all.append(par_seg)
                score_all.append(score)
            seq_all.append(c)  # seq_all = [ab, cde, f]
            seq = ['BOS']

    if len(score_all) != 0:
        score_mean = float(sum(score_all) / len(score_all))
    else:
        score_mean = 1.0

    return re.sub(r'\s+', '  ', '  '.join(seq_all).strip()), score_mean


def decode_file(testfile, segfile, window, max_word, sim, w2v, beam):
    print("Decoding file {}".format(testfile))
    n = 1
    with open(testfile, 'r') as inf, open(segfile, 'w') as ouf:
        for line in inf:
            line = line.strip()
            line = re.sub(r'\s+', '  ', line)
            seg, score = decode_split_by_punc(line, window, max_word, sim, w2v, beam)
            ouf.write(seg + '\n')
            print(n, seg, '%.3f' % score)
            n += 1
    print("Segmented text written in file {}".format(segfile))


if __name__ == "__main__":
    name = sys.argv[1]

    testfile = sys.argv[2]
    segfile = 'result/' + name + '_seg.txt'

    sim_path = 'src/' + name + '/sim.json'
    with open(sim_path, 'r') as f:
        sim = json.load(f)

    w2v_path = 'src/' + name + '/w2v.json'
    with open(w2v_path, 'r') as f:
        w2v = json.load(f)

    window = int(sys.argv[3])
    beam = int(sys.argv[4])
    max_word = int(sys.argv[5])

    decode_file(testfile, segfile, window, max_word, sim, w2v, beam)
