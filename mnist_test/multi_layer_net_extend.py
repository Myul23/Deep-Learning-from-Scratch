import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from collections import OrderedDict
from mnist_test.layers import *
from mnist_test.gradients import numerical_gradient

# 가중치 index는 W1, W2 이런 식으로 불러서 실제 인데스보다 하나 더 더한 값으로 부른다.
class MultiLayerNetExtend:
    """완전 연결 다층 신경망 (확장)
    가중치 감소, dropout, 배치 정규화 구현"""

    # ReLU: He 초깃값, Sigmoid or tanh: Xavier
    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
        use_dropout=False,
        dropout_rate=0.5,
        use_batchnorm=False,
    ):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        # 초기화
        self.params = {}
        self.__init_weight(weight_init_std)

        # (연산) 계층 쌓기
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)], self.params["b" + str(idx)])

            # 배치 정규화 층계 쌓기
            if self.use_batchnorm:
                self.params["gamma" + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params["beta" + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers["BatchNorm" + str(idx)] = BatchNormalization(self.params["gamma" + str(idx)], self.params["beta" + str(idx)])

            self.layers["Activation_function" + str(idx)] = activation_layer[activation]()

            # Dropout mask에 가까울 듯?
            if self.use_dropout:
                self.layers["Dropout" + str(idx)] = Dropout(dropout_rate)

        # 출력층을 위한 층계 나누기 & Softmax with Loss
        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)], self.params["b" + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std

            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params["W" + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        # 기울기 감소 구현
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = numerical_gradient(loss_W, self.params["W" + str(idx)])
            grads["b" + str(idx)] = numerical_gradient(loss_W, self.params["b" + str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads["gamma" + str(idx)] = numerical_gradient(loss_W, self.params["gamma" + str(idx)])
                grads["beta" + str(idx)] = numerical_gradient(loss_W, self.params["beta" + str(idx)])

        return grads

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = self.layers["Affine" + str(idx)].dW + self.weight_decay_lambda * self.params["W" + str(idx)]
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads["gamma" + str(idx)] = self.layers["BatchNorm" + str(idx)].dgamma
                grads["beta" + str(idx)] = self.layers["BatchNorm" + str(idx)].dbeta

        return grads
