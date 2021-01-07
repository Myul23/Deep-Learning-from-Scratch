import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import pickle
from mnist_test.mnist import load_mnist
from mnist_test.functions import sigmoid, softmax


def get_data():
    (trainX, trainY), (testX, testY) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return testX, testY


def init_network():
    with open(
            "C:\Github Projects\study_store\Deep Learning Projects\mnist_test\sample_weight.pkl",
            "rb") as f:
        # 일단 현재 위치로 되어있지만, 필요에 따라 Chapter 안으로 옮겨지면서 같이 옮긴다면 주소를 바꿔야 한다.
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + B3
    y = softmax(a3)
    return y
