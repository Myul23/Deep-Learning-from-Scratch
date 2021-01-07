# Chapter 3. 신경망

# 3.1 퍼셉트론에서 신경망으로
# 0과 1로 입력(및 출력)을 제한했던 퍼셉트론에서
# 가능한(상상할 수 있는) 모든 경우를 입력으로 하는 신경망으로의 개념 확대?

# 퍼셉트론과 신경망은 큰 차이를 보이지 않는다.
# 다만, 퍼셉트론은 임계값 처리를 통해 출력을 제어했다면
# 신경망은 은닉 노드에서 활성화 함수를 통해 출력을 제어한다.
# 신경망은 입력층과 출력층, 그리고 나머지의 은닉층으로 나뉘며
# 다음 층의 노드로 연결된 선이 각각의 가중치, 편향을 통한 선형 조합을 의미한다.

# 따라서, 단순 퍼셉트론은 임계값을 경계로 출력이 바뀌는 함수(계단 함수)를 이용하고
# 다층 퍼셉트론은 신경망(여러 층 구성 및 활성화 함수를 이용)을 가리킨다.


# 3.2 활성화 함수 (Activated Function)
# 선형은 아무리 층계를 많이 쌓아도 간결한(층이 없는) 선형으로 표현할 수 있다.
# 따라서, 비선형의 구현은 활성화 함수로 비선형을 사용해야 한다.

import numpy as np
import matplotlib.pyplot as plt

# 0. Step function
# def step_function(x):
#     if x > 0: return 1
#     else: return 0
# x의 형태와 관계없이 정수 하나만을 반환하므로 행렬에도 사용하기 위해 고칠 필요가 있다.

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.show()

# 1. Sigmoid: S자 모양 함수, 자연계수를 이용해 출력을 (0, 1)로 제한하는 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x)) # numpy 내부 broadcast가 잘 되어 있어 행렬도 가능하다

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# Step function has critical value on 0 and Sigmoid function is softer
# but they have 0 as minimum and 1 as maximum and are Non-linear activation functions.

# 2. ReLU (Rectified Linear Unit): 정류된 선형 함수, 일정(0) 이하의 값을 버린다(정류한다).
def relu(x):
    return np.maximum(0, x)


# 3.3 다차원 배열의 계산
A = np.array([1, 2, 3, 4])
print(A, np.ndim(A), A.shape, A.shape[0])
# .shape 배열 반환

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B, np.ndim(B), B.shape)

def dot_product(x, w, b = 0):
    if x.shape[-1] == y.shape[0]:
        return np.dot(x, y) + b
    else:
        return [A.shape, B.shape]

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(dot_product(A, B))

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
print(dot_product(A, B))

B = np.array([[1, 2], [3, 4]])
print(dot_product(A, B))

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])
print(dot_product(A, B))

# 신경망에서의 행렬 곱
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
print(dot_product(X, W))


# 3.4 3층 신경망 구현하기
# W<sup>n층의 가중치</sup><sub>다음 층 노드 번호, 앞 층 노드 번호</sub>
# 편향은 앞 층 노드 번호가 0으로 일정하므로 작성하지 않지만,
# 0번째 가중치로 가중치 뭉치에 묶일 때는 분리를 위해 작성하기도 한다.

# 첫번째 은닉층으로 비행 중
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
# (1, 2) x (2, 3) + (1, 3) = (1, 3)

# 첫번째 은닉층 (입국) 수속 밟기
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

# 두번째 은닉층으로 비행 중
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
# (1, 3) x (3, 2) + (1, 2) = (1, 2)

# 두번째 은닉층 (입국) 수속 밟기
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 출력층을 향해서
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
# (1, 2) x (2, 2) + (1, 2) = (1, 2)

A3 = np.dot(Z2, W3) + B3
Y = A3

# 한 방에 정리
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["B1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["B2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["B3"] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["B1"], network["B2"], network["B3"]

    a1 = np.dot(x, W1) + B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + B3
    y = a3
    return y

network = init_network()
y = forward(network, X)
print(y)


# 3.5 출력층 설계하기
# 회귀에선 항등함수를, 분류에선 소프트맥스를 출력층의 활성화함수로 이용한다.
# 그러나 확률을 원하는 것이 아니라 최대 확률값 클래스를 찾고자 한다면
# 지수 계산에 드는 자원을 절약하고자 원래 값에서 최댓값을 찾는 식으로 소프트맥스를 생략한다.

# softmax function: yk = exp(ak) / sum(exp(ai))
def softmax(a):
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    y = exp_a / np.sum(exp_a)
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y, np.sum(a))


# 3.6 손글씨 숫자 인식
import sys, os
sys.path.append(os.pardir)
from MNIST.mnist import load_mnist
(trainX, trainY), (testX, testY) = load_mnist(flatten=True, normalize=False)
# load_mnist: normalize(bool, 0 ~ 1), flatten, one_hot_label

from PIL import Image # Python Image Library

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    # numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환하는 과정
    pil_img.show()

img = trainX[0]; label = trainY[0]
print(label, img.shape)

img = img.reshape(28, 28)
print(img.shape)
img_show(img)
