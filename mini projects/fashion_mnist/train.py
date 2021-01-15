import sys, os, gzip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from mnist_test.trainer import Trainer
from sklearn.model_selection import train_test_split


def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


Xx, y_train = load_mnist("C:\\Github Projects\\study_store\\Deep Learning Projects\\mini projects\\fashion_mnist", kind="train")
Xt, y_test = load_mnist("C:\\Github Projects\\study_store\\Deep Learning Projects\\mini projects\\fashion_mnist", kind="t10k")
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


X_train = Xx.reshape(Xx.shape[0], 1, 28, 28)
X_test = Xt.reshape(Xt.shape[0], 1, 28, 28)

# X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2)
# print(
#     "train:",
#     X_train2.shape,
#     y_train2.shape,
#     "\n test:",
#     X_val.shape,
#     y_val.shape,
#     "\ntotal: ",
#     X_train2.shape[0] + X_val.shape[0],
#     "\t   /",
#     y_train2.shape[0] + y_val.shape[0],
# )

max_epochs = 20

network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01,
)

trainer = Trainer(
    network,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=max_epochs,
    mini_batch_size=100,
    optimizer="AdaGrad",  # optimizer="Adam",
    optimizer_param={"lr": 0.001},
    evaluate_sample_num_per_epoch=1000,
)
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {"train": "o", "test": "s"}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker="o", label="train", markevery=2)
plt.plot(x, trainer.test_acc_list, marker="s", label="test", markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()


y_pred = np.argmax(network.predict(X_train), axis=1)
print(network.accuracy(X_train, y_train))
y_pred = np.argmax(network.predict(X_test), axis=1)
print(network.accuracy(X_test, y_test))
