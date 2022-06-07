import os
import json
import pickle
from tqdm import tqdm
from itertools import chain
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer

base_path = os.path.join('umls_org', 'objects')

def chisquare_dataset(dev_x_idx, dev_y, top_words):
    """
    :param dataset: dev dataset
    :param top_words: max number of words with the highest Chi2 given each class
    :return: json object written in issu_description_path
    """
    issu_description = {}
    dev_x_set = set(dev_x_idx)

    dev_x_text = ['']*len(dev_x_idx)
    print('Loading training data for Chi2 compute ...')
    classes = list(set(chain(*dev_y)))

    for content_pack in tqdm(os.listdir(os.path.join(base_path, 'pmid2contents'))):
        pmid2mesh_terms_map = pickle.load(open(os.path.join(base_path, 'pmid2contents', content_pack), 'rb'))
        for pmid in pmid2mesh_terms_map:
            if pmid in dev_x_set:
                title = pmid2mesh_terms_map[pmid][1]
                abstract = pmid2mesh_terms_map[pmid][2]
                dev_x_text[dev_x_idx.index(pmid)] = '%s %s'%(title, abstract)

    print('Generating Chi2 dictionary...\n')
    for cls in tqdm(classes):
        y = []
        for labels in dev_y:
            if cls in labels:
                y.append(1)
            else:
                y.append(0)

        vectorizer = CountVectorizer(lowercase=True, stop_words='english')
        X = vectorizer.fit_transform(dev_x_text)

        chi2score = chi2(X,y)[0]
        wscores = zip(vectorizer.get_feature_names(),chi2score)
        wchi2 = sorted(wscores,key=lambda x:x[1])
        chi_words = [x[0] for x in wchi2[-top_words:][::-1]]
        issu_description[cls] = chi_words

    with open(os.path.join(base_path, 'class_chi2_words_path.json'), 'w') as wr:
        json.dump(issu_description, wr, indent=1)


if __name__ == "__main__":
    X_dev, y_dev = pickle.load(open(os.path.join(base_path, 'dataset', 'dev_ids.pkl'), 'rb'))
    chisquare_dataset(X_dev, y_dev, top_words=100)