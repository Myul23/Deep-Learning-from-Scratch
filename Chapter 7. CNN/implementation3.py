import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import numpy as np
from collections import OrderedDict
from mnist_test.mnist import load_mnist
from mnist_test.layers import *
from mnist_test.gradients import numerical_gradient
from mnist_test.trainer import Trainer


class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 가중치 초기화
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], conv_param["stride"], conv_param["pad"])
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads["W" + str(idx)] = numerical_gradient(loss_w, self.params["W" + str(idx)])
            grads["b" + str(idx)] = numerical_gradient(loss_w, self.params["b" + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i + 1)]
            self.layers[key].b = self.params["b" + str(i + 1)]


# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# # 시간이 오래 걸릴 경우 데이터를 줄인다.
# # x_train, t_train = x_train[:5000], t_train[:5000]
# # x_test, t_test = x_test[:1000], t_test[:1000]

# max_epochs = 20

# network = SimpleConvNet(
#     input_dim=(1, 28, 28),
#     conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
#     hidden_size=100,
#     output_size=10,
#     weight_init_std=0.01,
# )

# trainer = Trainer(
#     network,
#     x_train,
#     t_train,
#     x_test,
#     t_test,
#     epochs=max_epochs,
#     mini_batch_size=100,
#     optimizer="Adam",
#     optimizer_param={"lr": 0.001},
#     evaluate_sample_num_per_epoch=1000,
# )
# trainer.train()

# # 매개변수 보존
# network.save_params("params.pkl")
# print("Saved Network Parameters!")

import matplotlib.pyplot as plt

# # 그래프 그리기
# markers = {"train": "o", "test": "s"}
# x = np.arange(max_epochs)
# plt.plot(x, trainer.train_acc_list, marker="o", label="train", markevery=2)
# plt.plot(x, trainer.test_acc_list, marker="s", label="test", markevery=2)
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc="lower right")
# plt.show()


# Visualization
def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


def filter_show2(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, FS = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params["W1"])

# 학습된 가중치
network.load_params("params.pkl")
filter_show(network.params["W1"])

# fig = plt.figure()
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(20):
#     ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(network.params["W2"][i * 10].reshape(10, 10), cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()

# print(network.params["W3"].shape)
# filter_show2(network.params["W3"])
