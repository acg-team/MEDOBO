import os
import sys
import time
import json
import pickle
import random
import warnings
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
from itertools import chain
from keras.models import Model
from keras import backend as K
from tools import onehot_target, clean_str
from chi_sqaure import chisquare_dataset
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D


config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 8
sess = tf.Session(config=config)
K.set_session(sess)


emb_root = 'Resources'

base_path = os.path.join('umls_org', 'objects')

warnings.filterwarnings('ignore')
random.seed(1496)


class Ontology_classify():
    def __init__(self, num_training, max_features, emb_size):
        print('Initialization ...')
        self.num_training = num_training
        self.max_features = max_features
        self.emb_size = emb_size
        self.batch_size = 128
        self.epochs = 20
        self.hidden_size = 128
        self.drop_out = 0.5
        self.patience = 3
        self.issu_description_path = os.path.join(base_path, 'class_chi2_words_path.json')

    def load_data(self):
        print('Loading data ...')
        X_train, y_train = pickle.load(open(os.path.join(base_path, 'dataset', 'train_ids.pkl'), 'rb'))
        X_dev, y_dev = pickle.load(open(os.path.join(base_path, 'dataset', 'dev_ids.pkl'), 'rb'))
        X_test, y_test = pickle.load(open(os.path.join(base_path, 'dataset', 'test_ids.pkl'), 'rb'))
        X_train, y_train = X_train[:self.num_training], y_train[:self.num_training]
        if not os.path.isfile(self.issu_description_path):
            chisquare_dataset(X_dev, y_dev, top_words=100)

        descriptions = json.load(open(self.issu_description_path, 'r'))
        self.all_chi_words = set(chain(*[x for x in descriptions.values()]))

        train_idx, dev_idx, test_idx = set(X_train), set(X_dev), set(X_test)
        all_idx = train_idx.union(dev_idx).union(test_idx)
        self.X_train_txt, self.X_dev_txt, self.X_test_txt = [], [], []
        self.X_train_chi, self.X_dev_chi, self.X_test_chi = [], [], []
        self.y_train, self.y_dev, self.y_test = [], [], []
        print('\nLoading data ...')
        content_packs = [mesh_pack for mesh_pack in os.listdir(os.path.join(base_path, 'pmid2contents'))]
        for content_pack in tqdm(content_packs):
            pmid2content_map = pickle.load(open(os.path.join(base_path, 'pmid2contents', content_pack), 'rb'))
            for pmid, content in pmid2content_map.items():
                if pmid not in all_idx:continue
                title_abstract = clean_str('%s %s' % (content[0], content[1]))
                if pmid in train_idx:
                    self.X_train_txt.append(title_abstract)
                    self.X_train_chi.append(' '.join([cw for cw in title_abstract.split() if cw in self.all_chi_words]))
                    self.y_train.append(y_train[X_train.index(pmid)])
                elif pmid in dev_idx:
                    self.X_dev_txt.append(title_abstract)
                    self.X_dev_chi.append(' '.join([cw for cw in title_abstract.split() if cw in self.all_chi_words]))
                    self.y_dev.append(y_dev[X_dev.index(pmid)])
                elif pmid in test_idx:
                    self.X_test_txt.append(title_abstract)
                    self.X_test_chi.append(' '.join([cw for cw in title_abstract.split() if cw in self.all_chi_words]))
                    self.y_test.append(y_test[X_test.index(pmid)])
            pmid2content_map.clear()

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.y_train + self.y_dev)
        self.y_train = self.mlb.transform(self.y_train)
        self.y_dev = self.mlb.transform(self.y_dev)
        self.y_test = self.mlb.transform(self.y_test)


        print('Train/Dev/Tes/Num_Classes t: %d / %d / %d / %d' % (len(self.X_train_txt), len(self.X_dev_txt), len(self.X_test_txt), len(list(self.mlb.classes_))))

    def tokenize(self):
        print('Tokenization and Padding ...')
        self.tokenizer = text.Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(self.X_train_txt + self.X_dev_txt)
        self.seq_length_txt = int(np.mean([len(x.split()) for x in self.X_dev_txt]))
        self.seq_length_chi = int(np.mean([len(x) for x in self.X_dev_chi]))
        print('Mean Text sequence length: ', self.seq_length_txt)
        print('Mean Chi sequence length: ', self.seq_length_chi)

        self.X_train_txt = self.tokenizer.texts_to_sequences(self.X_train_txt)
        self.X_train_txt = sequence.pad_sequences(self.X_train_txt, maxlen=self.seq_length_txt)

        self.X_dev_txt = self.tokenizer.texts_to_sequences(self.X_dev_txt)
        self.X_dev_txt = sequence.pad_sequences(self.X_dev_txt, maxlen=self.seq_length_txt)

        self.X_test_txt = self.tokenizer.texts_to_sequences(self.X_test_txt)
        self.X_test_txt = sequence.pad_sequences(self.X_test_txt, maxlen=self.seq_length_txt)

        self.X_train_chi = self.tokenizer.texts_to_sequences(self.X_train_chi)
        self.X_train_chi = sequence.pad_sequences(self.X_train_chi, maxlen=self.seq_length_chi)

        self.X_dev_chi = self.tokenizer.texts_to_sequences(self.X_dev_chi)
        self.X_dev_chi = sequence.pad_sequences(self.X_dev_chi, maxlen=self.seq_length_chi)

        self.X_test_chi = self.tokenizer.texts_to_sequences(self.X_test_chi)
        self.X_test_chi = sequence.pad_sequences(self.X_test_chi, maxlen=self.seq_length_chi)

        self.nb_words = min(self.max_features, len(self.tokenizer.word_index))

    def embeddings(self):
        print('Loading the embedding matrix ...')
        lWords = open(emb_root + "/word2vecTools/types.txt", encoding="utf-8").readlines()
        lVectors = open(emb_root + "/word2vecTools/vectors.txt", encoding="utf-8").readlines()

        assert len(lVectors) == len(lWords)
        embeddings_index = {word.strip().lower():np.array([float(num) for num in lVectors[n].strip().split(" ")]) for n, word in enumerate(lWords)}
        self.embedding_matrix = np.zeros((self.nb_words, self.emb_size))

        unknown = 0
        for word, i in self.tokenizer.word_index.items():
            if i >= self.nb_words: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
            else:
                unknown += 1

        print('Embeddings shape: ', self.embedding_matrix.shape)

    def get_model(self):
        print('Initializing model ...')
        txt_input = Input(shape=(self.seq_length_txt, ))
        x1 = Embedding(self.max_features, self.emb_size, weights=[self.embedding_matrix])(txt_input)
        x1 = SpatialDropout1D(self.drop_out)(x1)
        x1 = Bidirectional(GRU(self.hidden_size, return_sequences=True))(x1)

        txt_avg_pool = GlobalAveragePooling1D()(x1)
        txt_max_pool = GlobalMaxPooling1D()(x1)

        chi_input = Input(shape=(self.seq_length_chi,))
        x2 = Embedding(self.max_features, self.emb_size, weights=[self.embedding_matrix])(chi_input)
        x2 = SpatialDropout1D(self.drop_out)(x2)
        x2 = Bidirectional(GRU(self.hidden_size, return_sequences=True))(x2)

        chi_avg_pool = GlobalAveragePooling1D()(x2)
        chi_max_pool = GlobalMaxPooling1D()(x2)

        conc = concatenate([txt_avg_pool, txt_max_pool, chi_avg_pool, chi_max_pool])

        outp = Dense(len(list(self.mlb.classes_)), activation="sigmoid")(conc)

        self.model = Model(inputs=[txt_input, chi_input], outputs=outp)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def report_result(self, y_pred, y_test):
        y_pred_one_hot5 = onehot_target(np.copy(y_pred), threshhold=0.5)

        print(classification_report(y_test, y_pred_one_hot5, target_names=list(self.mlb.classes_)))

    def train(self):
        print('Model Training ...')
        this_epoch_loss = 1000
        patience = 0
        t1 = time()
        for ep in range(self.epochs):
            if patience > self.patience:
                print('No improve after %d iteration. Training stops'%self.patience)
                break
            print('Training Model at epoch: ', ep)
            hist = self.model.fit([self.X_train_txt, self.X_train_chi], self.y_train, batch_size=self.batch_size, epochs=1, validation_data=([self.X_dev_txt, self.X_dev_chi], self.y_dev), verbose=0)

            if hist.history['val_loss'][0] < this_epoch_loss:
                patience = 0
                this_epoch_loss = hist.history['val_loss'][0]
                t10 = time()
                y_pred = self.model.predict([self.X_test_txt, self.X_test_chi], self.batch_size, verbose=0)
                t11 = time()
                infer_time = t11 - t10
                print('TRAIN TIME: ', infer_time)
                self.report_result(y_pred, self.y_test)
            else:
                patience +=1
        t2 = time()
        train_time = t2 - t1
        print('TRAIN TIME: ', train_time)


if __name__=="__main__":
    num_training = int(sys.argv[1])
    max_features =int(sys.argv[2])
    ont_classifier = Ontology_classify(num_training, max_features, emb_size=200)
    ont_classifier.load_data()
    ont_classifier.tokenize()
    ont_classifier.embeddings()
    ont_classifier.get_model()
    ont_classifier.train()















