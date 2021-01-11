import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from mnist_test.neuralnet import *
from mnist_test.two_layer_net import TwoLayerNet

(trainX, trainY), (testX, testY) = load_mnist(normalize=True,
                                              one_hot_label=True)
network = TwoLayerNet(inputs=784, hiddens=50, outputs=10)

# first
"""x_batch = trainX[:3]
y_batch = trainY[:3]

grad_numerical = network.numerical_gradient(x_batch, y_batch)
grad_backprop = network.gradient(x_batch, y_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ": ", str(diff))"""

# second
# hyper parameter
iters = 10000
trains = trainX.shape[0]
batch_size = 100
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(trains / batch_size, 1)

for i in range(iters):
    batch_mask = np.random.choice(trains, batch_size)
    x_batch = trainX[batch_mask]
    y_batch = trainY[batch_mask]

    grad = network.gradient(x_batch, y_batch)
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(trainX, trainY)
        test_acc = network.accuracy(testX, testY)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
