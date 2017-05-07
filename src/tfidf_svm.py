import os
import sys
import unicodedata
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC


def get_w(w, tag):
    if tag:
        return w.split('/')[0]
    else:
        return w


tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))


def remove_punctuation(text):
    return text.translate(tbl)


def pre_sentences(sentences):
    result = list()
    for sen in sentences:
        sen0 = unicode(sen, 'utf-8')
        sen1 = remove_punctuation(sen0)
        sen2 = sen1.lower()
        result.append(sen2)
    return result


if __name__ == '__main__':
    tag = False
    datadir = './SA2016-training_data'
    filenames = ['SA-training_positive.txt',
                 'SA-training_negative.txt',
                 'SA-training_neutral.txt',]
    # datadir = './SA2016-training-data-ws'
    # filenames = ['train_positive_tokenized.txt',
    #              'train_negative_tokenized.txt',
    #              'train_neutral_tokenized.txt',]
    label_codes = ['pos', 'neg', 'neutral']

    sentences = []
    labels = []
    for i, filename in enumerate(filenames):
        path = os.path.join(datadir, filename)
        label = label_codes[i]
        f = open(path, 'r')
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            words = [get_w(w, tag) for w in line.split()]
            sentences.append(' '.join(words))
            labels.append(label)

    sentences = pre_sentences(sentences)
    clf = LinearSVC()
    y = np.array(labels)
    count_vect = CountVectorizer()
    X_count = count_vect.fit_transform(sentences)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_count)
    X = X_tfidf
    predicted = cross_validation.cross_val_predict(clf, X, y, cv=5)
    print(metrics.classification_report(y, predicted))