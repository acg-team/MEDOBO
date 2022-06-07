import re

def clean_str(string):
    if string is None: string = 'None'
    string = re.sub(r"[^A-Za-z0-9\'\.\?]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def onehot_target(d, threshhold=0.5):
    """
    :param d: an array of probabilities of issues
    :return: a one-hot array of classes
    """
    d[d >= threshhold] = 1
    d[d < threshhold] = 0
    return d