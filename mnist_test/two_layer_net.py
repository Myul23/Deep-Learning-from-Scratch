import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from mnist_test.functions import *
from mnist_test.gradients import numerical_gradient as ng
from mnist_test.gradients import gradient_descent as gd


class TwoLayerNet:
    def __init__(self, inputs, hiddens, outputs, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.rand(inputs, hiddens)
        self.params["b1"] = np.zeros(hiddens)
        self.params["W2"] = weight_init_std * np.random.randn(hiddens, outputs)
        self.params["b2"] = np.zeros(outputs)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = ng(loss_W, self.params["W1"])
        grads["b1"] = ng(loss_W, self.params["b1"])
        grads["W2"] = ng(loss_W, self.params["W2"])
        grads["b2"] = ng(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = gd(loss_W, self.params["W1"])
        grads["b1"] = gd(loss_W, self.params["b1"])
        grads["W2"] = gd(loss_W, self.params["W2"])
        grads["b2"] = gd(loss_W, self.params["b2"])

        return grads
