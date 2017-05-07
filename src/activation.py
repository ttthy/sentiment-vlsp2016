# coding: utf-8
import numpy as np


class Sigmoid(object):
    @staticmethod
    def activate(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def derivative(a):
        return a * (1. - a)

    @staticmethod
    def getName():
        return 'sigmoid'


class Tanh(object):
    @staticmethod
    def activate(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivative(a):
        return 1. - (a * a)

    @staticmethod
    def getName():
        return 'tanh'


class RectifiedLinear(object):
    @staticmethod
    def activate(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def derivative(a):
        a[a > 0] = 1
        return a

    @staticmethod
    def getName():
        return 'rectified_linear'


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    e /= np.sum(e, axis=1, keepdims=True)
    return e