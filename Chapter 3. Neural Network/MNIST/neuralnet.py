import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from MNIST.mnist import load_mnist
from MNIST.activation_functions import sigmoid, softmax

def get_data():
    (trainX, trainY), (testX, testY) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return testX, testY

def init_network():
    with open("C:\Github Projects\study_store\Deep Learning Projects\Chapter 3. Neural Network\MNIST\sample_weight.pkl", "rb") as f:
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
