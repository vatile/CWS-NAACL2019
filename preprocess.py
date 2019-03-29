import json
from collections import defaultdict
import sys
import os


def build_w2i_w2c(infile, out_w2i, out_w2c):
    print('Building w2i and w2c files from {}'.format(infile))
    from keras.preprocessing.text import Tokenizer
    tok = Tokenizer(num_words=None,
                    filters='\t\n',
                    lower=False,
                    split=' ',
                    char_level=False)
    with open(infile, 'r') as inf:
        texts = inf.readlines()
    tok.fit_on_texts(texts)
    w2i = tok.word_index
    with open(out_w2i, 'w') as ouf:
        json.dump(w2i, ouf)
    w2c = tok.word_counts
    with open(out_w2c, 'w') as wcf:
        json.dump(w2c, wcf)
    print('w2i and w2c files successfully stored in {} and {}'.format(out_w2i, out_w2c))
    return w2i


def build_cooccur(infile, out_coo, window):
    print('Building co-occurrence file from {}'.format(infile))
    dic = defaultdict(int)
    with open(infile, 'r') as inf:
        for line in inf:
            line = line.split()
            line.reverse()
            for i in range(len(line)):
                for j in range(1, min(window, len(line) - i)):
                    dic['|'.join(sorted([line[i], line[i + j]]))] += 1  #
    with open(out_coo, 'w') as ouf:
        json.dump(dic, ouf)
    print('coo file successfully stored in {}'.format(out_w2i))
    return dic


if __name__ == "__main__":
    src_fold = 'src/' + sys.argv[1]
    if not os.path.isdir(src_fold):
        os.mkdir(src_fold)
    out_w2i = src_fold + '/w2i.json'
    out_w2c = src_fold + '/w2c.json'

    infile = sys.argv[2]

    build_w2i_w2c(infile, out_w2i, out_w2c)

    out_coo = src_fold + '/coo.json'
    window = int(sys.argv[3])
    build_cooccur(infile, out_coo, window)





