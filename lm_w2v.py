from keras.initializers import RandomNormal
from keras.layers import Input, Dense
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import skip_gram


class Word2Vec:
    """ A word2vec model implemented in keras. This is an implementation that
    uses negative sampling and skipgrams. """

    def __init__(self, path):
        self.model = None
        self.model_path = path

    def build(self, vector_dim, vocab_size, lr):
        """ returns a word2vec model """
        print("Building keras model...")

        stddev = 1.0 / vector_dim
        print("Setting initializer standard deviation to: {}".format(stddev))
        initializer = RandomNormal(mean=0.0, stddev=stddev, seed=10)

        word_input = Input(shape=(1,), name="word_input")
        context_input = Input(shape=(1,), name="context_input")

        Ebd = Embedding(input_dim=vocab_size + 1, output_dim=vector_dim, name="embedding",
                        embeddings_initializer=initializer)
        word = Ebd(word_input)
        context = Ebd(context_input)

        merged = dot([word, context], axes=2, normalize=True, name="cos")
        merged = Flatten()(merged)
        output = Dense(1, activation='sigmoid', name="output")(merged)

        optimizer = SGD(lr=lr)
        model = Model(inputs=[word_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def train(self, text, window_size, negative_samples, max_neg, sampling_table, neg_sampling_table, sampling_fix,
              neg_self, batch_size, w2i, w2c, epoch, val_split, shuffle):
        """ Trains the word2vec model """
        print("Training model...")

        seed = 1
        vocabulary_size = len(w2i)

        print("Preparing data...")

        couples, labels, pos_num, neg_num = skip_gram.skipgram_fix(text, vocabulary_size, w2i, w2c, max_neg,
                                                                   window_size, negative_samples, shuffle,
                                                                   False, sampling_table, neg_sampling_table,
                                                                   sampling_fix, neg_self, seed)
        print("Data loaded!")
        # in order to balance out more negative samples than positive
        try:
            negative_weight = pos_num / neg_num
        except Exception as e:
            print(e)
            print('No samples generated duo to too little training data!')
        negative_weight = (negative_weight + 0.2) / 1.2
        class_weight = {1: 1.0, 0: negative_weight}
        print("Class weights set to: {}".format(class_weight))

        checkpoint = ModelCheckpoint(self.model_path, monitor='loss', save_weights_only=False, save_best_only=True,
                                     verbose=2, mode='auto')

        earlystop = EarlyStopping(monitor='loss', patience=2, verbose=2, mode='auto')

        self.model.fit([couples[0], couples[1]], labels,
                       batch_size=batch_size,
                       epochs=epoch,
                       callbacks=[checkpoint, earlystop],
                       validation_split=val_split,
                       class_weight=class_weight)
        print("Model successfully trained and saved in {}".format(self.model_path))
