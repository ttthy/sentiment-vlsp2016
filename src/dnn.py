# coding: utf-8
import numpy as np
import time
import activation as act
import cPickle


def getCrossEntropyCost(a, y):
    # J = np.nan_to_num(- 1. * y * np.log(a))
    j = - 1. * y * np.log(a)
    cost = j.mean()
    return cost


class MLNN(object):
    def __init__(self, layersizes, fn):
        self.layersizes = layersizes
        self.B = [np.zeros((x,)) for x in layersizes[1:]]
        self.W = [np.random.randn(x, y) / x
                  for x, y in zip(layersizes[:-1], layersizes[1:])]
        self.fn = fn
        print ('Layersizes: ' + str(self.layersizes))
        print ('Activate function: ' + self.fn.getName())

    def setLayersizes(self, layersizes):
        self.layersizes = layersizes
        self.B = [np.zeros((x,)) for x in layersizes[1:]]
        self.W = [np.random.randn(x, y) / x
                  for x, y in zip(layersizes[:-1], layersizes[1:])]

    def setFn(self, fn):
        self.fn = fn

    def setParams(self, B, W):
        self.B = B[:]
        self.W = W[:]

    def getParams(self):
        return (self.B, self.W)

    def feedforward(self, a):
        iteration = len(self.layersizes) - 2
        for i in range(iteration):
            a = self.fn.activate(a.dot(self.W[i]) + self.B[i])
        a = act.softmax(a.dot(self.W[-1]) + self.B[-1])
        return a

    def computeError(self, x, y):
        predict_y = self.feedforward(x)
        predict_labels = np.argmax(predict_y, axis=1)
        return np.mean(predict_labels != y)

    def computeAcc(self, x, y):
        predict_y = self.feedforward(x)
        predict_labels = np.argmax(predict_y, axis=1)
        return np.mean(predict_labels == y)

    def getCost(self, x, y):
        a = self.feedforward(x)
        cost = getCrossEntropyCost(a, y)
        return cost

    def backprop(self, x, y):
        grad_B = [np.zeros(b.shape) for b in self.B]
        grad_W = [np.zeros(w.shape) for w in self.W]

        # ------- Feed forward
        activation = x
        activations = [x]

        for b, w in zip(self.B[:-1], self.W[:-1]):
            activation = self.fn.activate(activation.dot(w) + b)
            activations.append(activation)

        # output layer
        activation = act.softmax(activation.dot(self.W[-1]) + self.B[-1])
        activations.append(activation)

        # ------- Backward
        delta = activations[-1] - y  # cross entropy error
        grad_B[-1] = np.sum(delta, axis=0)
        grad_W[-1] = activations[-2].T.dot(delta)

        for l in xrange(2, len(self.layersizes)):
            delta = delta.dot(self.W[-l + 1].T) * self.fn.derivative(activations[-l])
            grad_B[-l] = np.sum(delta, axis=0)
            grad_W[-l] = activations[-l - 1].T.dot(delta)

        return (grad_B, grad_W)

    def SGD(self, x, y, dev_x, dev_y, max_epoch, max_patience, mnb_size, lr, lda, model='model.pkl'):
        n_samples = x.shape[0]
        one_hot_y = np.zeros((n_samples, 3))
        rand_idic = range(n_samples)

        one_hot_y[rand_idic, y] = 1.

        costs = []
        a_in = []
        a_dev = [0]
        best_pos = 0
        patience = max_patience

        print 'Begin SGD....'
        epoch = 0
        while (epoch < max_epoch):
            start = time.time()
            np.random.shuffle(rand_idic)

            for start_idx in range(0, n_samples, mnb_size):
                mnb_x = x[rand_idic[start_idx:start_idx + mnb_size]]
                mnb_y = one_hot_y[rand_idic[start_idx:start_idx + mnb_size]]
                delta_b, delta_w = self.backprop(mnb_x, mnb_y)
                self.W = [w - lr * (dw / mnb_size + lda * w)
                          for w, dw in zip(self.W, delta_w)]
                self.B = [b - (lr / mnb_size) * db
                          for b, db in zip(self.B, delta_b)]

            costs.append(self.getCost(x, one_hot_y))
            a_in.append(self.computeAcc(x, y))
            a_dev.append(self.computeAcc(dev_x, dev_y))

            end = time.time()

            if a_dev[-1] < a_dev[best_pos]:
                patience -= 1
            else:
                patience = max_patience
                best_pos = len(a_dev) - 1
                name = 'vlsp-mnb5-lda01-lr1-'
                with open(name+'.pkl', 'wb') as f:
                    cPickle.dump((self.B, self.W), f)

            # print "Epoch %d, %fs/epoch, training acc %f, val acc %f, patience %d" % (
            # epoch, end - start, a_in[-1], a_dev[-1], patience)

            if patience == 0:
                break
            epoch += 1

        print 'End training...'
        return (costs, a_in, a_dev)