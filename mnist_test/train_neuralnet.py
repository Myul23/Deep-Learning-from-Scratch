import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from mnist_test.mnist import load_mnist
from mnist_test.two_layer_net import TwoLayerNet

(trainX, trainY), (testX, testY) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iters = 1000
trains = trainX.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(inputs=784, hiddens=50, outputs=10)
for i in range(iters):
    batch_mask = np.random.choice(trains, batch_size)
    x_batch = trainX[batch_mask]
    y_batch = trainY[batch_mask]

    # grad = network.numerical_gradient(x_batch, y_batch)
    grad = network.gradient(x_batch, y_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)
# iter_num과 Cost function의 함수를 보여주자.
# 계속 학습하면서 loss 값이 줄어들고 있음.

# 4.5.3 시험 데이터로 평가하기
# 앞선 loss는 훈련 데이터에 대한 오차 지표고 더 믿음직한 오류율을 구하고자
# 드디어 test 데이터를 사욯합니다.
# addition) epoch: 학습에서 훈련 데이터를 모두 소진했을 때의 횟수
# 10,000개를 100개의 미니배치로 학습하면 SGD를 100회 반복하면서 모든 훈련 데이터를 소진하게 됨. 이때 100회가 1에폭
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from mnist_test.mnist import load_mnist
from mnist_test.two_layer_net import TwoLayerNet

(trainX, trainY), (testX, testY) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(inputs=784, hiddens=50, outputs=10)

iters = 1000
trains = trainX.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(trains / batch_size, 1)

for i in range(iters):
    batch_mask = np.random.choice(trains, batch_size)
    x_batch = trainX[batch_mask]
    y_batch = trainY[batch_mask]

    # grad = network.numerical_gradient(x_batch, y_batch)
    grad = network.gradient(x_batch, y_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    # 정확도 계산 per 1 epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(trainX, trainY)
        test_acc = network.accuracy(testX, testY)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("train acc, test acc : " + str(train_acc) + ", " + str(test_acc))
