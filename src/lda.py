from reader import readtokens
from gensim import corpora, models
import gensim
from numpy import random
from time import time



if __name__ == '__main__':
    datadir = './SA2016-training-data-ws'
    filenames = ['train_positive_tokenized.txt',
                 'train_negative_tokenized.txt',
                 'train_neutral_tokenized.txt', ]

    start = time()
    sentences, labels = readtokens(datadir, filenames)
    random.shuffle(sentences)
    print (len(sentences))
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(sentences[:4500])
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in sentences[:4500]]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

    topics = ldamodel[sentences[4500]]
    end = time()

    print (topics)
    print ('LDA time: {}'.format(end - start))