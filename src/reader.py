import os
from time import time

def readfile(datadir, filenames):
    """

    :param dirpath:
    :param filenames:
    :return:
    """
    sentences = list()
    start = time()
    labels = list()
    for i, file in enumerate(filenames):
        filepath = os.path.join(datadir, file)
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '': continue
                labels.append(i)
                sentences.append(line)
    end = time()
    print ("Time of loading...{}".format(end - start))
    return sentences, labels


def readtokens(datadir, filenames):
    """

    :param dirpath:
    :param filenames:
    :return:
    """
    sentences = list()
    start = time()
    labels = list()
    for i, file in enumerate(filenames):
        filepath = os.path.join(datadir, file)
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '': continue
                labels.append(i)
                sentences.append(line.split())
    end = time()
    print ("Time of loading...{}".format(end - start))
    return sentences, labels

if __name__ == '__main__':
    datadir = './SA2016-training-data-ws'
    filenames = ['train_positive_tokenized.txt',
                 'train_negative_tokenized.txt',
                 'train_neutral_tokenized.txt', ]
    readtokens(datadir, filenames)