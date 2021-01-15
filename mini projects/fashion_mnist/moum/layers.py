# 깔끔한 코드를 위한 layer class 분리
# tanh는 구현하지 않았음.
import numpy as np

# 입력 및 모형에 대하여
class Dropout:
    def __init__(self, dropout_ratio=0.1):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, input, train_flag=True):
        # 굳이 None임을 확인하는 게 아니라 flag를 준 건 dropout_rate에 대해 중간에 바꿨을 때를 대비한 것처럼 보인다.
        if train_flag:
            self.mask = np.random.rand(*input.shape) > self.dropout_ratio
            return input * self.mask
        else:
            return input * (1 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# 매개변수와의 곱: 행렬곱, 합성곱
class Affine:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.input = None
        self.dWeight = None
        self.dBias = None

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weight) + self.bias

    def backward(self, dout):
        self.dWeight = np.dot(self.input.T, dout)
        self.bias = np.sum(dout, axis=1)
        return np.dot(dout, self.weight.T)


# 그냥 분리해버린 CNN, Convolutional Neural Network
class CNN:
    def __init__(self):
        pass


# 마지막 계층에 대한 activation function격
def softmax(x):
    max_value = np.max(x)  # overflow 방지
    exp_x = np.exp(x - max_value)
    return exp_x / np.sum(exp_x)


# Activation functions
class Sigmoid:
    def __init__(self):
        self.out = None

    # 어째 out을 이용해야 함은 아는데, 저장할 생각을 안 하냐.
    def forward(self, input):
        self.out = 1 / (1 + np.exp(-input))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, input):
        self.mask = input <= 0
        input[self.mask] = 0
        return input

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


# Loss or Cost function
def gradient_descent(func, init, lr=0.01, step=100):
    x = init
    for i in range(step):
        # numerical gradient
        h = 1e-4
        grad = np.zeros_like(x)

        for i in range(x.size):
            temp = x[i]

            x[i] = temp + h
            func1 = func(x)
            x[i] = temp - h
            func2 = func(x)

            grad[i] = (func1 - func2) / (2 * h)
            x[i] = temp

        x -= lr * grad
    return x


# 식을 기억하는 건 좋은데, 벡터화해서 쓸 순 없는 걸까.
def cross_entropy_error(pred, test):
    if pred.ndim == 1:
        pred = pred.reshape(1, pred.size)
        test = test.reshape(1, test.size)

    # 미니배치로 이용하려고
    batch_size = pred.shape[0]
    return -np.sum(np.log(pred[np.arange(batch_size), test] + 1e-7)) / batch_size


# 이 계층은 나눠서는 잘하면서 왜 합치니까 못하냐.
class SoftmaxWithLoss:
    def __init__(self):
        # loss 확인 과정 때문에 남김
        self.loss = None
        self.pred = None
        self.test = None

    def forward(self, input, test):
        self.pred = softmax(input)
        self.test = test
        self.loss = cross_entropy_error(self.pred, self.test)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.test.shape[0]
        return (self.pred - self.test) / batch_size
