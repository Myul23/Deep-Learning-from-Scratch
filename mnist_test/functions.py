import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    y = exp_a / np.sum(exp_a)
    return y


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
