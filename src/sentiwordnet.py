import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import LinearSVC
import re
from time import time
from reader import readfile
from dnn import MLNN
from activation import Sigmoid, Tanh, RectifiedLinear
import cPickle


def get_senti_wordnet(filepath, headnum=0):
    """
    :param filepath:
    :param head_num:
    :return:
    """

    wordnet = dict()
    f = open(filepath, 'r')
    for i in range(headnum):
        f.readline()
    while True:
        line = f.readline().strip()
        if line == '': break
        tokens = line.split('\t')
        regex = r"(.+?)#\d+?"
        terms = re.findall(regex, tokens[4])
        for term in terms:
            term = term.strip(' ')
            if term in wordnet:
                raise KeyError('Duplicate term in wordnet')
            wordnet[term] = [float(tokens[2]), float(tokens[3])]
    f.close()
    return wordnet


def read_mywordnet(filepath, headnum=0, featurenum=3):
    """

    :param filepath: string, path to file contains wordnet
    :param headnum: int, number of headers
    :return:
    """
    wordnet = dict()
    i = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '': continue
            tokens = line.split('\t')
            pos = float(tokens[1])
            neg = float(tokens[2])
            if featurenum == 3:
                wordnet[tokens[0]] = [i, pos, neg, 1. - (pos + neg)]  # id, pos score, neg score, objectivity score
            elif featurenum == 2:
                wordnet[tokens[0]] = [i, pos, neg]
            elif featurenum == 1:
                wordnet[tokens[0]] = i
            i += 1
    return wordnet


def score_vectorize(sentence, wordnet, featindex=3):
    x = np.zeros(len(wordnet))
    for term in wordnet:
        if term in sentence:
            x[wordnet[term][0]] = wordnet[term][featindex]
    return x


def preprocess_wordnet():
    net = get_senti_wordnet('./VietSentiWordnet_ver1.0.txt', head_num=23)


def use_senti_wordnet(datadir, filenames, wordnet, featindex=3):
    print ("Wordnet size: {}".format(len(wordnet)))

    start = time()
    X = list()
    sentences, labels = readfile(datadir, filenames)
    for sentence in sentences:
        X.append(score_vectorize(sentence, wordnet, featindex=featindex))
    del sentences
    labels = np.array(labels)
    X = np.array(X)
    end = time()
    print('Time for vectorization: {}'.format(end - start))
    return X, labels


def use_tfidf(datadir, filenames, min_df=1, vocabulary=None):
    if vocabulary:
        tfidf = TfidfVectorizer(min_df=min_df, vocabulary=vocabulary, ngram_range=(1, 4))
    else:
        tfidf = TfidfVectorizer(min_df=min_df)

    X = list()
    X, labels = readfile(datadir, filenames)
    labels = np.array(labels)
    X = tfidf.fit_transform(X)

    # store the content
    with open("tfidf-feature.pkl", 'wb') as handle:
        cPickle.dump(tfidf, handle)
    return X, labels


def use_bow(datadir, filenames, min_df=1, vocabulary=None):
    if vocabulary:
        bow = CountVectorizer(min_df=min_df, vocabulary=vocabulary, ngram_range=(1, 4))
    else:
        bow = CountVectorizer(min_df=min_df)

    X = list()
    X, labels = readfile(datadir, filenames)
    labels = np.array(labels)
    X = bow.fit_transform(X)

    # store the content
    with open("bow-feature.pkl", 'wb') as handle:
        cPickle.dump(bow, handle)
    return X, labels


def demo():
    datadir = './SA2016-training-data-ws'
    filenames = ['train_positive_tokenized.txt',
                 'train_negative_tokenized.txt',
                 'train_neutral_tokenized.txt', ]
    vi_senti_wordnet = 'visenti.txt'
    # use tf-idf as features
    # X, y = use_bow(datadir, filenames, min_df=5)
    X, y = use_tfidf(datadir, filenames, min_df=5)
    # use tf-idf of vietsentiwordnet
    # senti_vocab = read_mywordnet(vi_senti_wordnet, featurenum=1)
    # X, y = use_bow(datadir, filenames, vocabulary=senti_vocab)
    # X, y = use_tfidf(datadir, filenames, vocabulary=senti_vocab)
    # use vietsentiwordnet as features
    # senti_vocab = read_mywordnet(vi_senti_wordnet)
    # X, y = use_senti_wordnet(datadir, filenames, senti_vocab, featindex=2)


    # clf = LinearSVC()
    # predicted = cross_validation.cross_val_predict(clf, X, y, cv=10)
    # print(metrics.classification_report(y, predicted))

    # Simple K-Fold cross validation. 10 folds.
    cv = cross_validation.KFold(X.shape[0], n_folds=10, shuffle=True)

    precision, recall, f1score = [], [], []
    for traincv, testcv in cv:
        # SVM
        # preds = clf.fit(X[traincv], y[traincv]).predict(X[testcv])
        #NN
        X_train, X_dev, y_train, y_dev = cross_validation.train_test_split(X[traincv], y[traincv], test_size=0.1)
        nn = MLNN([X_train.shape[1], 100, 3], RectifiedLinear())
        nn.SGD(X_train, y_train, X_dev, y_dev, max_epoch=100, max_patience=20, mnb_size=5, lr=0.1, lda=0.01)
        print X_train.shape, X_dev.shape
        # set params of NN at stage of highest validation acc
        name = 'vlsp-mnb5-lda01-lr1-'
        with open(name+'.pkl', 'rb') as f:
            B, W = cPickle.load(f)
        nn.setParams(B, W)
        # load tfidf from train data
        # pos - neg - neu
        preds = nn.feedforward(X[testcv])
        preds = np.argmax(preds, axis=1)
        p, r, f1, s = metrics.precision_recall_fscore_support(y[testcv], preds,
                                                      labels=[0,1,2],
                                                      average=None)
        precision.append(p)
        recall.append(r)
        f1score.append(f1)

    print "\tprecision\trecall\tf1score"
    plabel = np.mean(precision, axis=0)
    rlabel = np.mean(recall, axis=0)
    f1label = np.mean(f1score, axis=0)
    print "0\t{:.2f}\t{:.2f}\t{:.2f}\n".format(plabel[0], rlabel[0], f1label[0])
    print "1\t{:.2f}\t{:.2f}\t{:.2f}\n".format(plabel[1], rlabel[1], f1label[1])
    print "2\t{:.2f}\t{:.2f}\t{:.2f}\n".format(plabel[2], rlabel[2], f1label[2])
    print "\t{:.2f}\t{:.2f}\t{:.2f}\n".format(np.mean(precision), np.mean(recall), np.mean(f1score))
    # print "precision {}".format(precision)
    # print "recall {}".format(recall)
    # print "f1score {}".format(f1score)

    # load test data
    # lb = ['POS','NEG','NEU']
    # X_test = list()
    # X_test, _ = readfile('T:/CTT/Data/Sentiment/VLSP2016/SA2016-test', ['test_tokenized.txt'])
    # X_test = tfidf.transform(X_test)
    # print (X_test.shape)
    # predicted_y = nn.feedforward(X_test)
    # predicted = np.argmax(predicted_y, axis=1)
    # print predicted
    # del predicted_y
    # with open(name+'.re', 'w') as f:
    #     for _y in predicted:
    #         f.write("{}\n".format(lb[_y]))



if __name__ == '__main__':
    demo()
    # X_test, _ = readfile('T:/CTT/Data/Sentiment/VLSP2016/SA2016-test', ['test_tokenized.txt'])
    # lab, _ = readfile('T:/CTT/Projects/VLSP', ['vlsp-mnb10-lda01-lr1.re'])
    # print lab
    # with open('T:/CTT/Projects/VLSP/result.txt', 'w') as f:
    #     for i in range(len(lab)):
    #         f.write("{}\n{}\n".format(X_test[i], lab[i]))
