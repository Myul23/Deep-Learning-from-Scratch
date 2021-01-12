import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from mnist_test.mnist import load_mnist
from mnist_test.utils import shuffle_dataset

(x_train, t_train), (x_test, t_test) = load_mnist()

# validation을 rand mask를 통해 뽑는 방법도 있지만, 여기선 전체 데이터를 shuffle하는 방식을 채택.
x_train, t_train = shuffle_dataset(x_train, t_train)

x_train = x_train[:500]
t_train = t_train[:500]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

# Optimization
import matplotlib.pyplot as plt
from mnist_test.multi_layer_net import MultiLayerNet
from mnist_test.trainer import Trainer


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay
    )

    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=epocs,
        mini_batch_size=100,
        optimizer="sgd",
        optimizer_param={"lr": lr},
        verbose=False,
    )
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# hyper-paramter 최적화 과정
optimization_trial = 100
results_val = {}
results_train = {}

# iter_num과 관계없이 그냥 반복할 거라서.
for _ in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    val_acc_list, train_acc_list = __train(lr, weight_decay)

    key = "lr: " + str(lr) + ", weight decay: " + str(weight_decay)
    print("val acc: " + str(val_acc_list[-1]) + " | " + key)

    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

print("============== Hyper-Parameter Optimization Result ==============")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

# key: sorting 기준점
for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    print("Best-" + str(i + 1) + "(val acc: " + str(val_acc_list[-1]) + ") | " + key)

    x = np.arange(len(val_acc_list))

    plt.subplot(row_num, col_num, i + 1)
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")

    plt.ylim(0.0, 1.0)
    # y는 모든 그래프가 축 값을 보여주면 지저분하니까 첫줄만(false) 보여주도록 한 거였음.
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    plt.title("Best-" + str(i + 1))

    i += 1
    if i >= graph_draw_num:
        break

plt.show()
# 학습률이 높은 순으로 나열된 걸 보면서 그 다음은 어떻게 하자라고 조정.
# 현재 테스트 데이터는 lr값이 0.009까지 커질 때 정확도를 크게 줌.
