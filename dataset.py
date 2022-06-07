
import os
import json
import pickle
import shutil
import obonet
from tqdm import tqdm
from ftplib import FTP
from time import sleep
import pubmed_parser as pp
from urllib import request
from random import shuffle
from dateutil import parser
from itertools import chain
from multiprocessing import Pool
from collections import defaultdict
from sklearn.model_selection import train_test_split

num_workers = 10
base_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'

base = 'umls'
mrconso_path = os.path.join(base, 'META/MRCONSO.RRF')
mrxns_eng_path = os.path.join(base, 'META/MRXNS_ENG.RRF')
mrxnw_eng_path = os.path.join(base, 'META/MRXNW_ENG.RRF')
mrsty_path = os.path.join(base, 'META/MRSTY.RRF')

base_ontology_folder = 'obo_verified'
base_path = os.path.join(base, 'objects')
new_folders = ['medline_xml', 'pmid2contents', 'pmid2mesh_expands', 'pmid2labels', 'dataset']

for nf in new_folders:
    if not os.path.isdir(os.path.join(base_path, nf)):
        os.makedirs(os.path.join(base_path, nf), exist_ok=True)

medline_path = os.path.join(base_path, 'medline_xml')


def clean_title(title):
    title = ' '.join(title) if isinstance(title, list) else title

    if title.startswith('['):title = title[1:]
    if title.endswith(']'): title = title[:-1]
    if title.endswith('.'): title = title[:-1]
    if title.endswith(']'): title = title[:-1]
    return title.lower() + ' .'


def clean_abstract(abstract):
    if abstract.endswith('.'): abstract = abstract[:-1] + ' .'
    return abstract.lower()


def get_medline_files_path():
    """
    :return: helper function to get medline file names
    """
    file_names = []
    with FTP('ftp.ncbi.nlm.nih.gov') as ftp:
        ftp.login()
        lines = []
        ftp.dir('pubmed/baseline', lines.append)
        for i in lines:
            tokens = i.split()
            name = tokens[-1]
            time_str = tokens[5] + " " + tokens[6] + " " + tokens[7]
            modified = str(parser.parse(time_str))
            size = tokens[4]

            if name.endswith('.gz'):
                file_names.append(name)
    return file_names


def medline_download(renew=False):
    print('Downloading Medline XML files ...')
    file_names = get_medline_files_path()
    for f_name in tqdm(file_names):
        if not os.path.isfile(os.path.join(medline_path, f_name)) or renew:
            if f_name not in os.listdir(medline_path):
                with request.urlopen(os.path.join(base_url, f_name)) as response, open(os.path.join(medline_path, f_name), 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                    sleep(1)


def medline_parser(med_xml):
    dicts_out = pp.parse_medline_xml(os.path.join(medline_path, med_xml),
                                     year_info_only=False,
                                     nlm_category=False,
                                     author_list=False,
                                     reference_list=False)

    pack = []
    for i in dicts_out:
        pmid = i['pmid']
        c_title = clean_title(i['title'])
        title = c_title if len(c_title)>10 else None

        c_abstract = clean_abstract(i['abstract'])
        abstract = c_abstract if len(c_abstract)>10 else None

        if len(i['mesh_terms']):
            mesh_terms = [x.strip().split(':')[1].lower() for x in i['mesh_terms'].split(';')]
        else:
            mesh_terms = None

        if all([title, abstract, mesh_terms]):
            pack.append((pmid, title, abstract, mesh_terms))
    return pack


def multi_process_medline(renew=False):
    """
    :return: list of tuples where pmids are mapped to their mesh terms, titles and abstracts (strings)
    """
    print('Processing XML files ...')
    if len(os.listdir(os.path.join(base_path, 'pmid2contents'))) and renew is False:return
    xml_files = [xml_file for xml_file in os.listdir(medline_path) if xml_file.endswith('.xml.gz')]
    shuffle(xml_files) #load balance files with different sizes
    for idx in tqdm(range(0, len(xml_files), 10)):
        xml_files_batch = xml_files[idx: idx + 10]
        with Pool(processes=num_workers) as pool:
            pmid2content_map_all = pool.map(medline_parser, xml_files_batch)
        pmid2content_map_all = list(chain(*pmid2content_map_all))

        pmid2content = defaultdict(set)
        for entry in pmid2content_map_all:
            pmid2content[entry[0]] = entry[1:]

        with open(os.path.join(base_path, 'pmid2contents', 'pmid2content%d.pkl' % idx), 'wb') as f:
            pickle.dump(pmid2content, f)
        pmid2content.clear()


def cui2ui(umls_mrconso_path):
    """
    :param umls_mrconso_path:
    :return: map each mesh major heading cui to list of lui and sui
    """
    print('Loading CUI to UI mappings ...')
    obj_path = os.path.join(base_path, 'cui2ui_map.pkl')
    if 'cui2ui_map.pkl' in os.listdir(base_path):
        cui2ui_map= pickle.load(open(obj_path, 'rb'))
    else:
        cui2ui_map  = defaultdict(set)
        with open(umls_mrconso_path, 'r') as f:
            for line in f:
                splits = line.strip().split('|')
                assert len(splits) == 19
                cui = splits[0]
                lat = splits[1]
                lui = splits[3]
                sui = splits[5]
                sab = splits[11]
                tty = splits[12]
                if lat =='ENG' and sab=='MSH' and tty=='MH':
                    cui2ui_map[cui].add(lui)
                    cui2ui_map[cui].add(sui)
        with open(obj_path, 'wb') as w:
            pickle.dump(cui2ui_map, w)
    return cui2ui_map


def pref_term2cui(umls_mrconso_path):
    """
    :param umls_mrconso_path:
    :return: mapping preferred terms to cuis:
    Each cui (if limited to Major Heading Mesh terms) has only one preferred term
    """
    print('Loading Preferred terms to CUI mappings ...')
    obj_path = os.path.join(base_path, 'pref_term2cui_map.pkl')
    if os.path.isfile(obj_path):
        pref_term2cui_map = pickle.load(open(obj_path, 'rb'))
    else:
        pref_term2cui_map = {} # cuis limited to Major Heading Mesh terms have only one preferred term(otherwise, 2.11)
        with open(umls_mrconso_path, 'r') as f:
            for line in f:
                splits = line.strip().split('|')
                assert len(splits) == 19
                cui = splits[0]
                lat = splits[1]
                sab = splits[11]
                tty = splits[12]
                pref_term = splits[14].lower()
                if lat =='ENG' and sab=='MSH' and tty=='MH':
                    pref_term2cui_map[pref_term] = cui
        with open(obj_path, 'wb') as wr:
            pickle.dump(pref_term2cui_map, wr)
    return pref_term2cui_map


def ui2string(umls_norm_string_path, umls_norm_word_path, valid_uis):
    """
    :param umls_norm_string_path:
    :param umls_norm_word_path:
    :param valid_uis:
    :return: given a ui(lui, sui) when Mesh term mapped to cui and cuis mapped to ui get all associated strings
    """
    ui2string_map  = defaultdict(set)
    for source_file_path in [umls_norm_string_path, umls_norm_word_path]:
        with open(source_file_path, 'r') as f:
            for line in f:
                splits = line.strip().split('|')
                assert len(splits) == 6
                strng = splits[1].lower()
                lui = splits[3]
                sui = splits[4]
                if lui in valid_uis:
                    ui2string_map[lui].add(strng)
                if sui in valid_uis:
                    ui2string_map[sui].add(strng)
    return ui2string_map



def expand_mesh(content_pack):
    pmid2mesh_terms_map = pickle.load(open(os.path.join(base_path, 'pmid2contents', content_pack), 'rb'))
    batch_number = content_pack.split('.')[0].replace('pmid2content', '')
    pmid2ui_map = defaultdict(set)
    pmid2expanded_mesh_map = defaultdict(set)

    valid_uis = set()
    for pmid in pmid2mesh_terms_map:
        mesh_terms = pmid2mesh_terms_map[pmid][-1]
        if mesh_terms:
            for mesh_term in mesh_terms:
                if mesh_term in pref_term2cui_map:
                    mesh_cui = pref_term2cui_map[mesh_term]
                    for ui in cui2ui_map[mesh_cui]:
                        pmid2ui_map[pmid].add(ui)
                        valid_uis.add(ui)
    # transforming uis to their string representations
    ui2string_map = ui2string(mrxns_eng_path, mrxnw_eng_path, valid_uis)
    for pmid, uis in pmid2ui_map.items():
        for ui in uis:
            if ui in ui2string_map:
                for st in ui2string_map[ui]:
                    pmid2expanded_mesh_map[pmid].add(st)
        for mesh_term in pmid2mesh_terms_map[pmid][-1]: # add mesh strings themselves in addition to the expanded lui and sui
            pmid2expanded_mesh_map[pmid].add(mesh_term)
    obj_path = os.path.join(base_path, 'pmid2mesh_expands', 'pmid2mesh_expand')
    with open('%s%s.pkl'%(obj_path, batch_number), 'wb') as wr:
        pickle.dump(pmid2expanded_mesh_map, wr)

    ui2string_map.clear()
    valid_uis.clear()
    pmid2ui_map.clear()
    pmid2expanded_mesh_map.clear()


def multi_process_expand_mesh(renew=False):
    print('Expanding Mesh mappings ...')
    if len(os.listdir(os.path.join(base_path, 'pmid2mesh_expands'))) and renew is False:return

    xml_packs = [xml_pack for xml_pack in os.listdir(os.path.join(base_path, 'pmid2contents'))]

    with Pool(processes=num_workers) as pool:
        pool.map(expand_mesh, xml_packs)


def get_ontology_mapping(ontology_folder):
    """
    :param ontology_folder:
    :return: provided ontologies in obo format, returns the inverted indexes for exact matching
    """
    print('Compiling ontology inverted indexes ...')
    ontology_mappings = {}
    for ont_name in tqdm(os.listdir((ontology_folder))):
        obo_ont_path = os.path.join(ontology_folder, ont_name)
        graph = obonet.read_obo(obo_ont_path)
        name2id = {}
        for id_, data in graph.nodes(data=True):
            if data.get('name'):
                name2id[data.get('name').lower()] = id_
            if data.get('synonym'):
                synonyms = [x.split("RELATED")[0].strip().replace('"', '').replace("'", '') for x in data.get('synonym') if 'RELATED' in x]
                for syn in synonyms:
                    name2id[syn] = id_
        ontology_mappings[ont_name] = name2id
    return ontology_mappings


def get_ontology_assignments(mesh_pack):
    pmid2ontology = defaultdict(list)
    batch_number = mesh_pack.split('.')[0].replace('pmid2mesh_expand', '')
    pmid2expanded_strings_map = pickle.load(open(os.path.join(base_path, 'pmid2mesh_expands', mesh_pack), 'rb'))
    for pmid, mesh_strings in pmid2expanded_strings_map.items():
        ontology_matches = {}
        for ontology_name, ontology_mapping in ontology_mappings.items():
            if len(mesh_strings):
                ontology_matches[ontology_name.replace('.obo', '').upper()] = sum([1 for mt in mesh_strings if mt in ontology_mapping])/len(mesh_strings)
        ontology_matches = {k:v for k,v in ontology_matches.items() if v>0}
        pmid2ontology[pmid] = list(sorted(ontology_matches.items(), key=lambda x:x[1], reverse=True))

    pmid2expanded_strings_map.clear()
    with open(os.path.join(base_path, 'pmid2labels', 'pmid2label%s.pkl'%batch_number), 'wb') as wr:
        pickle.dump(pmid2ontology, wr)
    pmid2ontology.clear()


def multi_process_ontology_assignments(renew=False):
    print('Ontology assignment ...')
    if len(os.listdir(os.path.join(base_path, 'pmid2labels'))) and renew is False: return

    mesh_packs = [mesh_pack for mesh_pack in os.listdir(os.path.join(base_path, 'pmid2mesh_expands'))]

    with Pool(processes=num_workers) as pool:
        pool.map(get_ontology_assignments, mesh_packs)

def dataset():
    print('Compiling data splits ...')
    pmids, labels = [], []
    for pmid2label in tqdm(os.listdir(os.path.join(base_path, 'pmid2labels'))):
        pmid2label_pack = pickle.load(open(os.path.join(base_path, 'pmid2labels', pmid2label), 'rb'))
        for pmid, label in pmid2label_pack.items():
            if pmid not in pmids_with_duplicate_labels:
                pmids.append(pmid)
                labels.append([x[0] for x in label if x[1]>=0.10])

    X_tr_dv, X_test, y_tr_dv, y_test = train_test_split(pmids, labels, test_size=100000, random_state=100, shuffle=True)
    X_train, X_dev, y_train, y_dev = train_test_split(X_tr_dv, y_tr_dv, test_size=100000, random_state=100, shuffle=True)

    with open(os.path.join(base_path, 'dataset', 'train_ids.pkl'), 'wb') as wr:
        pickle.dump([X_train, y_train], wr)

    with open(os.path.join(base_path,  'dataset', 'dev_ids.pkl'), 'wb') as wr:
        pickle.dump([X_dev, y_dev], wr)

    with open(os.path.join(base_path,  'dataset', 'test_ids.pkl'), 'wb') as wr:
        pickle.dump([X_test, y_test], wr)


def main():
    medline_download()
    multi_process_medline()
    multi_process_expand_mesh()
    multi_process_ontology_assignments()
    dataset()


if __name__ == "__main__":
    cui2ui_map = cui2ui(mrconso_path)
    pref_term2cui_map = pref_term2cui(mrconso_path)
    ontology_mappings = get_ontology_mapping(base_ontology_folder)
    pmids_with_duplicate_labels = json.load(open('pmids_with_duplicate_labels.json', 'r'))
    main()