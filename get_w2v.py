from lm_w2v import Word2Vec
import json
import sys
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm


def get_text(infile):
    text = []
    with open(infile, 'r') as inf:
        for line in inf:
            line = line.strip()
            if line != '':
                text.append(line.split(' '))
    return text


def get_w2v(model_path, w2i, w2v_path):
    print("Loading trained embeddings...")
    model = load_model(model_path)
    ebd = model.layers[2].get_weights()[0]
    w2v_dic = {}
    w2v_dic_json = {}
    for w in w2i:
        w2v_dic[w] = ebd[w2i[w]].reshape(1, -1)
        w2v_dic_json[w] = [float(i) for i in ebd[w2i[w]]]
    with open(w2v_path, 'w') as f:
        json.dump(w2v_dic_json, f)
    print("Trained embeddings loaded and stored in {}".format(w2v_path))
    return w2v_dic


def store_sim(w2v, coo, sim_path):
    print("Storing cosine similarity of co-occurring word pairs ...")
    sim_dic = {}
    for keys in tqdm(coo):
        x_1, x_2 = w2v[keys.split('|')[0]], w2v[keys.split('|')[1]]
        sim_dic[keys] = float(cos_sim(x_1, x_2))
    with open(sim_path, 'w') as f:
        json.dump(sim_dic, f)
    print("Cosine similarity stored in {}".format(sim_path))
    return sim_dic


if __name__ == "__main__":
    name = sys.argv[1]

    w2i_file = 'src/' + name + '/w2i.json'
    w2c_file = 'src/' + name + '/w2c.json'
    w2v_file = 'src/' + name + '/w2v.json'
    coo_file = 'src/' + name + '/coo.json'
    with open(w2i_file, 'r') as f:
        w2i = json.load(f)
    with open(w2c_file, 'r') as f:
        w2c = json.load(f)
    with open(coo_file, 'r') as f:
        coo = json.load(f)

    infile = sys.argv[2]
    text = get_text(infile)

    window_size = int(sys.argv[3])

    vector_dim = 100
    vocab_size = len(w2i)
    lr = 0.01
    negative_samples = 1
    max_neg = 10
    sampling_fix = True
    neg_self = True
    batch_size = 128
    epoch = 1
    sampling_table = 'zipf'
    neg_sampling_table = 'uniform'
    val_split = 0.1
    shuffle = True

    model_path = 'model/' + name + '.model'
    model = Word2Vec(model_path)
    model.build(vector_dim, vocab_size, lr)

    model.train(text, window_size, negative_samples, max_neg, sampling_table, neg_sampling_table, sampling_fix,
                neg_self, batch_size, w2i, w2c, epoch, val_split, shuffle)

    w2v = get_w2v(model_path, w2i, w2v_file)

    sim_path = 'src/' + name + '/sim.json'
    store_sim(w2v, coo, sim_path)
