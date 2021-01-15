import numpy as np
from layers import *


# 완전 연결
class fullyConnected:
    def __init__(self, learning_rate=0.01, weight_decay=0, dropout_ratio=0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_ratio = dropout_ratio

        self.layers = {}
        self.lastLayer = None

    def train(self, input, output, hidden_node_layer=[30], activation="relu"):
        act_func = {"sigmoid": Sigmoid, "relu": Relu}

        for index in range(len(hidden_node_layer)):
            self.layers["Affine" + str(index)] = Affine()
            self.layers[activation + str(index)] = act_func[activation]()

    def predict(self, test_input):
        inputs = test_input
        for i in range(len(self.layers - 1)):
            inputs = self.layers[i].forward(inputs)
        outputs = self.lastLayer.forward(inputs)
        return outputs

    def accuracy(self, test_input, test_labels):
        outputs = self.predict(test_input)
        return np.sum(outputs == test_labels) / 100
