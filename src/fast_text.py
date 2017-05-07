from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np


def use_senti_tfidf(vocab):
    sentences = list()
    datadir = './SA2016-training-data-ws'
    filenames = ['train_positive_tokenized.txt',
                 'train_negative_tokenized.txt',
                 'train_neutral_tokenized.txt', ]

    for i, file in enumerate(filenames):
        filepath = os.path.join(datadir, file)
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '': continue
                sentences.append(line)

    vect = TfidfVectorizer(min_df=1, vocabulary=vocab, ngram_range=(1, 4))
    x = vect.fit_transform(sentences)
    print (x)


def main():
    vocab = dict()
    index = 0

    with open('./visenti.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            vocab[tokens[0]] = index
            index += 1
    print len(vocab)
    # use_senti_tfidf(vocab)


def prepross_fasttext(datadir, filenames):
    for i, file in enumerate(filenames):
        filepath = os.path.join(datadir, file)
        outfile = os.path.join(datadir, 'ft_{}'.format(file))
        label = '__label__{}'.format(i)
        with open(filepath, 'r') as f:
            with open(outfile, 'w') as w:
                for line in f:
                    line = line.strip()
                    if line == '': continue
                    w.write('{} {}\n'.format(label, line))


def divide_data(datadir, filenames):
    for i, file in enumerate(filenames):
        filepath = os.path.join(datadir, file)
        outfile = os.path.join(datadir, 'd_ft_{}'.format(file))
        testfile = os.path.join(datadir, 'd_ft_test_{}'.format(file))
        label = '__label__{}'.format(i)
        count = 0
        with open(filepath, 'r') as f:
            with open(outfile, 'w') as w:
                with open(testfile, 'w') as t:
                    for line in f:
                        line = line.strip()
                        if line == '': continue
                        if count == 1530:
                            t.write('{} {}\n'.format(label, line))
                        else:
                            w.write('{} {}\n'.format(label, line))
                            count += 1

def count_tokens(datadir, filenames):
    dic = {}
    for i, file in enumerate(filenames):
        filepath = os.path.join(datadir, file)
        with open(filepath, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                for w in tokens:
                    if w not in dic:
                        dic[w] = 1
    print len(dic)


def count_ngrams(path):
    count = [[], [], [], [], []]
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            count[tokens[0].count(' ')] += [tokens[0]]
    print count


if __name__ == '__main__':
    # main()
    datadir = './SA2016-training-data-ws'
    filenames = ['train_positive_tokenized.txt',
                 'train_negative_tokenized.txt',
                 'train_neutral_tokenized.txt', ]
    # prepross_fasttext(datadir, filenames)
    count_tokens(datadir='/home/thy/Downloads', filenames=['train_s.txt'])
    # divide_data(datadir, filenames)
    # count_ngrams('./visenti.txt')
    