import os
import sys
import json
import pickle
import random
import warnings
import numpy as np
from tqdm import tqdm
from time import time
from tools import clean_str
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


base_path = os.path.join('umls_org', 'objects')

warnings.filterwarnings('ignore')

random.seed(1496)


def data(ont, X_train, y_train, X_test, y_test):
    x_train_data_plus = [x for n, x in enumerate(X_train) if ont in y_train[n]]
    y_train_data_plus = [1 for _ in x_train_data_plus]

    x_train_data_minus = [x for n, x in enumerate(X_train) if ont not in y_train[n]]
    y_train_data_minus = [0 for _ in x_train_data_minus]

    x_test_data_plus = [x for n, x in enumerate(X_test) if ont in y_test[n]]
    y_test_data_plus = [1 for _ in x_test_data_plus]

    x_test_data_minus = [x for n, x in enumerate(X_test) if ont not in y_test[n]]
    y_test_data_minus = [0 for _ in x_test_data_minus]

    x_train = x_train_data_plus + x_train_data_minus
    y_train = np.array(y_train_data_plus + y_train_data_minus)

    x_test = x_test_data_plus + x_test_data_minus
    y_test = np.array(y_test_data_plus + y_test_data_minus)

    return x_train, y_train, x_test, y_test


class Ontology_classify():
    def __init__(self, num_training):
        print('Initialization ...')
        self.num_training = num_training
        descriptions = json.load(open(os.path.join(base_path, 'class_chi2_words_path.json'), 'r'))
        self.ontologies = list(descriptions.keys())


    def load_data(self):
        print('Loading data ...')
        X_train, y_train = pickle.load(open(os.path.join(base_path, 'dataset', 'train_ids.pkl'), 'rb'))
        X_dev, y_dev = pickle.load(open(os.path.join(base_path, 'dataset', 'dev_ids.pkl'), 'rb'))
        X_test, y_test = pickle.load(open(os.path.join(base_path, 'dataset', 'test_ids.pkl'), 'rb'))
        X_train, y_train = X_train[:self.num_training], y_train[:self.num_training]

        X_test, y_test = X_test[:self.num_training], y_test[:self.num_training]
        X_dev, y_dev = X_dev[:self.num_training], y_dev[:self.num_training]


        train_idx, dev_idx, test_idx = set(X_train), set(X_dev), set(X_test)
        all_idx = train_idx.union(dev_idx).union(test_idx)
        self.X_train_txt, self.X_dev_txt, self.X_test_txt = [], [], []
        self.X_train_chi, self.X_dev_chi, self.X_test_chi = [], [], []
        self.y_train, self.y_dev, self.y_test = [], [], []
        content_packs = [mesh_pack for mesh_pack in os.listdir(os.path.join(base_path, 'pmid2contents'))]
        for content_pack in tqdm(content_packs):
            pmid2content_map = pickle.load(open(os.path.join(base_path, 'pmid2contents', content_pack), 'rb'))
            for pmid, content in pmid2content_map.items():
                if pmid not in all_idx:continue
                title_abstract = clean_str('%s %s' % (content[0], content[1]))
                if pmid in train_idx:
                    self.X_train_txt.append(title_abstract)
                    self.y_train.append(y_train[X_train.index(pmid)])

                elif pmid in dev_idx:
                    self.X_dev_txt.append(title_abstract)
                    self.y_dev.append(y_dev[X_dev.index(pmid)])

                elif pmid in test_idx:
                    self.X_test_txt.append(title_abstract)
                    self.y_test.append(y_test[X_test.index(pmid)])
            pmid2content_map.clear()


if __name__=="__main__":
    num_training = int(sys.argv[1])
    ont_classifier = Ontology_classify(num_training)
    ont_classifier.load_data()

    f1s, recalls, perss = [], [], []
    tp, tn, fp, fn = [], [], [], []
    train_times, infer_times = [], []
    for ont in tqdm(ont_classifier.ontologies):
        x_train, y_train, x_test, y_test = data(ont, ont_classifier.X_train_txt, ont_classifier.y_train, ont_classifier.X_test_txt, ont_classifier.y_test)

        parameters = {'tfidf__use_idf': (True,),}

        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))), ('tfidf', TfidfTransformer(use_idf=True)), ('clf', MultinomialNB(alpha=1e-2))])
        t1 = time()
        clf.fit(x_train, y_train)
        t2 = time()
        train_time = t2 - t1
        predicted = clf.predict(x_test)
        t3 = time()
        infer_time = t3 - t2
        train_times.append(train_time)
        infer_times.append(infer_time)


        for i in zip(y_test, predicted):
            if i[0] == i[1] == 1:
                tp.append(1)
            if i[0] == i[1] == 0:
                tn.append(1)
            if i[0] != i[1] and i[1] == 1:
                fn.append(1)
            if i[0] != i[1] and i[1] == 0:
                fp.append(1)
    recall = sum(tp) / float(sum(tp) + sum(fn))
    percision = sum(tp) / float(sum(tp) + sum(fp))
    f1 = (2 * recall * percision) / float(percision + recall)

    f1s.append(f1)
    recalls.append(recall)
    perss.append(percision)
    print('\ntp:%d, tn:%d, fp:%d, fn:%d' % (sum(tp), sum(tn), sum(fp), sum(fn)))
    print('\np:%0.2f, r:%0.2f, f1:%0.2f' % (percision, recall, f1))
    print("Time train/infer: %d / %d"%(sum(train_times), sum(infer_times)))







