from __future__ import print_function, division
# from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T*np.log(Y)).sum()


def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def y2indicator(y): # what does this do?
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]]  #invalid indexing
    return ind


def get_training_data(balance_ones=True):
    dir = '/Users/rhinomonkey/Desktop/MLFolder'

    raw_images = np.loadtxt('1.txt')  # converts to numpyarray
    labels = np.loadtxt('3.txt')  # converts to numpyarray

    X = [] #result labels
    Y = [] #result images

    #print (np.shape(raw_images))

    for line in raw_images:
        b = np.array(line, dtype=np.float)
        #b = np.reshape(b, (-1, 12))
        b = (b / b.max())
        Y.append(b) #result_images
        #img = Image.fromarray(b, 'L')
        #img.show() #shows image combo
        #sys.exit()

    for line in labels:
        X.append(line[1]) #result_labels

    Y = np.array(Y)


    #return np.array(Y), np.array(X)
    return  Y, X

#X, Y = get_training_data()


def getData_tmp():
    dir = '/Users/rhinomonkey/Desktop/MLFolder'
    raw_images = np.loadtxt('1.txt')  # converts to numpyarray
    labels = np.loadtxt('3.txt')  # converts to numpyarray
    for x0 in raw_images:
        x0 *=1.0/x0.max()
    X, Y = np.array(raw_images), np.array(labels[:,1])
    #set index here

    return X, Y

def getImageData():
    X, Y = getData_tmp()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def crossValidation(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) // K
    scores = []
    for k in range(K):
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        model.fit(xtr, ytr)
        score = model.score(xte, yte)
        scores.append(score)
    return scores

def crossValidation_tf(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) // K
    scores = []
    for k in range(K):
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        score = model.fit(xtr, ytr, xte, yte)
        scores.append(score)
    return scores


