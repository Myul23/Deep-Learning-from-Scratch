# Chapter 4. 신경망 학습

import numpy as np
import matplotlib.pyplot as plt

# 4.1 데이터에서 학습한다.
# 데이터를 보고 가중치의 값을 (컴퓨터가 or 알아서) 계산한다.
# 퍼셉트론 수렴 정리: 퍼셉트론도 선형 분리 문제에 대해선 유한 번의 학습으로 가중치를 조절할 수 있다.

# 4.1.1 데이터 주도 학습
# 인간의 사고에는 경험과 직관이라는 접근 방식이 있음. 그러나 매번 사람이 많은 양의 데이터를 (전체적으로) 훑어서 (초기) 접근 방식을 찾을 순 없음.
# 맞닥드린 상황에 대한 경험 부족이 이유일 수도 있고, 단순히 데이터가 많기 때문일 수도 있음.
# 그래서 경험과 직관 이외에 사람이 문제 풀이에 접근하는 방식 중 하나인 규칙(특징) 찾기를 통해 데이터에 접근하고자 함.
# 마치 5는 중간 아래쪽에 타원에 가까운 곡선이 있고, 왼쪽 아래와 오른쪽 위에 선에 끝이 있는 등의 특징
# 여기서 특징이란 입력 데이터가 가진 본질적인 데이터를 정확하게 추출할 수 있도록 설계된 변환기를 의미
# 다시 말해 앞서 말한 곡선이나 선이 끝나는 지점 등을 찾아 이러한 특징이 있는지 확인하는지를 의미?
# 이미지의 특징은 보통 벡터로 (변환)기술하고, CV에선 SIFT, SURF, HOG 등의 특징을 많이 이용
# 그 다음에 대표적인 지도 학습 분류 방법인 SVM, KNN 등을 사용
# 신경망은 더 나아가 5의 특징을 사람이 정의하지 않는 것.
# 그래서 딥러닝(신경망)을 종단간 기계학습 (end-to-end machine learning)이라고도 함.
# 결과적으로 신경망은 모든 문제에 대해 주어진 데이터만을 활용해 ene-to-end로 학습할 수 있음.

# 4.1.2 훈련 데이터와 시험 데이터
# 입력한 데이터를 확실하게 분리해내면 좋을 것 같음.
# 그러나 우리가 만들어내는 모델은 우리가 입력한 데이터 말고 다른 데이터에도 제대로 작동(분류, 판단)해야 함.
# 가령 예를 들어 MNIST 데이터처럼 손글씨 숫자 판별 모델을 만들었는데
# 1은 짝대기만 긋는 사람도 있고, serif?체로 부분을 다 살리는 사람도 있고, 7도 마찬가지.
# 모델의 학습 데이터로 제 손 글씨만 사용했다면, 다른 사람의 손 글씨의 1과 7은 구분이 어려울 것.
# 이렇게 하나의 데이터셋, 학습에 이용한 데이터셋에에 지나치게 학습률, 다시 지나치게 낮은 오류율을 가지는 상태를 오버피팅, 과대적합이라고 함.
# 학습한 데이터만 완벽하게 분리해내는 시스템은 사실상 필요가 없음.
# 그래서 이런 과대적합을 수치적으로 평가하고자 준비한 데이터에서 일부를 test data로 남겨둠.
# 이 데이터는 학습을 위해 사용하는 데이터가 아님.

# 4.2 손실 함수
# 들어가기에 앞서 지금부터 출력으로 예측된 값은 원-핫 인코딩을 통해 변환된 값임.
# 원-핫 인코딩은 n번째 클래스를 가리키는 값, n을 행렬에서 실제로 그 위치를 가리키도록 (벡터로) 변환하는 것을 말함.
# 다시 말해, n개의 클래스에서 4번째 클래스를 가리키는 값, 4는 [0, 0, 0, 0, 1, 0, 0, ..., 0]

# 앞서 데이터를 통해 학습한다는 의미는 신경망을 이루는 노드들의 가중치와 편향을 컴퓨터가 계산하는 거라고 했음.
# 이는 다시 말해 수치적으로 최적화된 값을 컴퓨터로 계산한다는 것이고,
# 이때 이를 계산하고 최적에 대한 평가 지표가 되는 함수를 손실 함수라 함.
# 다시 이 아이는 손실함수, Loss function으로 불리거나 비용함수, Cost function으로 불림.


# 4.2.1 오차제곱합
# 오차에 대한 수치적 지표로 사용되고, 선형식의 최적화된 모수를 찾는 방법 중, 가장 대표적인 것.
# 모델에 의해 예측된 값과 실제값의 차이를 제곱해 모두 합한 것.
# 이를 다시 코드로 표현하면
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t)**2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))

# 이런 식으로 실제값과의 차이에 대해 계산할 수 있음.

# 4.2.2 교차 엔트로피 오차
# 여기서 t는 실제 클래스 값이 원-핫 인코딩을 통해 변환된 값이고,
# 오차제곱합이 모든 클래스에 대해서 실제값과 해당 클래스일 확률에 대한 오차를 합한 거라면
# 교차 엔트로피 오차는 실제 클래스에 대해 해당 클래스일 확률만을 가지고 값을 계산한다.
# 다시, 실제로 2번째 클래스이고, 예측된(or 출력) 값이 이렇다면


# 2번째 클래스일 확률만을 가지고 로그연산한 값을 loss로 가진다.
# 이를 코드로 표현하면
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# 4.2.3 미니배치 학습
# 신경망이 추구하는 최적화는 손실 함수의 값이 가장 작은 매개변수들(가중치, 편향)값을 찾는 것.
# 그러나 경우를 다 계산하는 건 물질적, 시간적 자원 낭비 (이는 컴퓨터도 마찬가지)
# 당장 mnist 데이터만 봐도 600개라 for문 하나당 1초의 시간만 잡아도 N*K 초가 걸림.
# (물론 빅데이터 수준으로 입력 데이터가 확장되었는데도 최적화된 매개변수 값을 구하는데 5분이 안 걸린다면 사용할 의향 있음.)
# 그래서 데이터 일부를 골라 학습을 진행함.
# 이때 선택되는 일부를 미니배치라 부르고 훈련 데이터 중 일부만을 가지고 학습을 진행하는 것을 미니배치 학습이라 함.

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from mnist_test.mnist import load_mnist
(trainX, trainY), (testX, testY) = load_mnist(normalize=True,
                                              one_hot_label=True)
# train, test 데이터의 클래스 값을 원-핫 인코딩을 한 값을 가져오게 함.

# 해당 데이터에 미니배치 학습을 진행하면
train_size = trainX.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
batchX = trainX[batch_mask]
batchT = trainY[batch_mask]

# 보이는 것처럼 np.random.choice를 통해 랜덤으로 0부터 train_size까지의 수 중, batch_size만큼 고르는 것.


# 4.2.4 (배치용) 교차 엔트로피 오차 구현하기
# np.random.choice를 이용해 교차 엔트로피 오차를 다시 표현하면
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshpae(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 만약 원-핫 인코딩을 하지 않은 라벨이 들어온다면
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# y는 소프트맥스의 출력물이니까

# 4.2.5 왜 손실 함수를 설정하는가?
# 예컨대 신경망의 학습에서 최적의 값을 찾는 건 정확도가 높은 모델을 구성하기(찾기) 위함이고,
# 가중치의 최적의 값을 찾고자 오차에 대한 수치 지표를 미분해 이용함.
# 미분을 이용하는 이유는 미분은 fx에서 x에 따른 y값에 변화를 나타내고,
# 가중치 값의 변화에 따른 수치 지표의 변화를 나타내기 때문.
# 그런데 정확도가 지표가 되면 미분된 값이 0인 지점이 너무 많음.
# 다시 말해 정확도는 가중치 값에 대한 미분 가능한 함수가 아님.
# 그래서 이를 뒤집어 생각한 것. 정확도를 높이는 것에서 오차를 줄이는 것으로
# 그것도 손실함수라는 연산을 거쳐 수치 지표의 값을 정수에서 실수로 만들었음.
# 정확도에 대해 함수로 표현 -> 이는 계단함수 -> 앞서와 같은 이유로 계단함수 또한 신경망에서 사용하지 않음
# Sigmoid는 어떤 곳에서도 미분값이 완벽히 0이 되지 않음(수렴해서 0인 구간은 있음)


# 4.3 수치 미분 (numerical differentiation)
# 4.3.1 미분
# 앞서 말한 미분은 fx에서 x값에 의한 출력의 변화에 대한 정도로 표현
# 다시 말해 미분은 특정 순간에 출력의 변화 정도를 나타냄.
# 수식으로는 d/dx * f(x) = lim[f(x + h) - f(x) / h]
# 이를 코드로 구현하면
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) / h  # x가 앞이라 해서 전방 차분


# 더 작은 값을 이용하고 싶지만, 프로그래밍에는 반올림 오차라는 것이 존재.
# 너무 작은 값은 부동소수점에서 제대로 표기할 수 없어 버려진다고.
# 그리고 f(x + h) - f(x) / h는 x + h와 x 사이의 기울기를 구하는 것이 되므로 원래 구하려면 x의 위치는 아니게 됨.
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 의도적으로 중앙을 x로 맞춰줌. (이를 중앙 차분, 중심 차분이라 함.)
# 수치 미분: 아주 작은 차분으로 미분하는 것, 미분의 기본 개념에 따른 근사치 계산
# 해석적 미분: 이론적으로 원래 값을 계산, d(x^2)/dx = 2x


# 4.3.2 수치 미분의 예
def function_1(x):
    return 0.01 * (x**2) + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.plot(x, y)
plt.ylabel("f(x)")
plt.xlabel("x")
plt.show()

print(numerical_diff(function_1, 5), numerical_diff(function_1, 10), sep="\n")

# 해석적 미분의 값과 비교하면 매우 작은 차이가 남.


# 4.3.3 편미분
# 앞선 미분에선 변수가 하나였음. 그런데 편미분은 변수가 하나가 아님.
# 그렇지만, 크게 달라지는 건 없음. 미분하고자 하는 대상만을 변수로 취급하고 나머지는 상수 취급.
def function_2(x):
    return x[0]**2 + x[1]**2


def function_tmp1(x0):
    return x0 * x0 + 4.0**2.0


def function_tmp2(x1):
    return 3.0**2.0 + x1 * x1


print(numerical_diff(function_tmp1, 3.0),
      numerical_diff(function_tmp2, 4.0),
      sep="\n")


# 4.4 기울기
# 지금까지는 한 번에 하나의 변수로만 미분했음.
# 좀 더 편의를 위해 한 번에 하나 이상의 변수로 각각 미분을 진행하려고 함.
# 이때 미분된 값을 행렬로 묶어 표현한 것을 기울기라고 함.
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # zeros_like 설명을 살짝
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])),
      numerical_gradient(function_2, np.array([0.0, 2.0])),
      numerical_gradient(function_2, np.array([3.0, 0.0])),
      sep="\n")

# numpy의 가독성 좋은, 보기 좋은 출력에 대한 처리가 들어가서 정수로 보임.


# 4.4.1 경사법(경사하강법)
# 기계학습과 신경망은 학습 단계에서 최적의 가중치를 찾아야 합니다.
# 여기서 최적이란 손실 함수의 최솟값을 의미하고 이는 미분값을 통해 찾을 수 있습니다.
# ((-x)^2를 예로 들며) (-x)^2의 함수의 최솟값은 함수의 형태를 보고 구할 수도 있지만, 이를 미분을 통해 수치적으로 구할 수 있습니다.
# 미분값이 0이 되는 지점은 원래 함수의 변화량이 0이 되는 지점이고,
# 그 중에서 미분값 부호가 반전되는 지점은 원래 함수에서 최댓값 혹은 회솟값을 찍고 값이 반전되는 지점을 의미한다.
# 이렇게 미분, 기울기를 통해 손실 함수의 최솟값을 찾아가는 방법을 경사하강법이라 함.
# 다시 경사하강법은 초기 지점의 기울기(미분값)를 보고 일정 만큼 움직이고, 다시 기울기를 확인하고 움직이고를 반복하면서 손실 함수의 값을 점차 줄이는 것입니다.
# 이와 유사하지만, 손실 함수의 부호를 반전시켜 손실 함수의 최댓값을 구하기도 합니다.
# 부호의 반전이외에 딱히 달라지는 것은 없지만, 최솟값이 아닌 최댓값을 찾는다고 해서 경사 상승법이라고 부릅니다.
# 앞서 얘기한 경사하강법을 식으로 표현하면 이렇게 됩니다.
# 여기서 기울기를 확인하고 움직이는 크기를 학습률(learning rate)이라고 합니다.
# 학습률이 크면 크게 움직일 것이고, 작으면 아주 조금 값을 갱신하겠죠.
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# 앞서 정의한 function_2를 통해 경사하강법을 진행하면
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=0.1))
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=10))
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=1e-10))
# 학습률이 너무 작으면 가중치가 초기값에서 크게 변화하지 않고 끝나며, 너무 크면 최솟값으로 수혐하지 못하고 끝납니다.

# 4.4.2 신경망에서의 기울기
# 손실 함수에 대한 편미분 기울기는 가중치의 각 원소들로 편미분을 진행하기 때문에 가중치와 shape이 같습니다.
from mnist_test.simpleNet import *
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p, np.argmax(p), sep="\n")

t = np.array([0, 0, 1])
print(net.loss(x, t))

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)

# 4.5 학습 알고리즘 구현하기
# 신경망에는 입력층, 출력층 말고도 은닉층이 존재합니다. 이 은닉층은 노드로 구성되며 가중치와 편향을 이용합니다.
# 그러나 모든 데이터를 학습하는 것은 비효율적입니다, 그래서 임의의 표본, 미니 배치를 통해 모형 학습을 진행합니다.
# 최적의 값은 손실함수의 최솟값을 찾는 것으로 귀결되며 손실함수의 편미분을 통해 얻은 기울기를 이용합니다.
# 이번 단계의 기울기 계산을 마치면 학슴률과의 곱에 따라 기울기의 방향으로 이동합니다.(값을 갱신시킵니다)
# 이떄 미니배치는 무작위로 선정되기 때문에 이를 확률적 경사 하강법 줄여서 SGD라고 부릅니다.

# 4.5.1 2층 신경망 클래스 구현하기
# 적당히 클래스 설명, 손실 함수가 cross entropy라든지
from mnist_test.two_layer_net import *

net = TwoLayerNet(inputs=784, hiddens=100, outputs=10)
print(net.params["W1"].shape,
      net.params["b1"].shape,
      net.params["W2"].shape,
      net.params["b2"],
      sep="\n")

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
y = net.predict(x)

grads = net.numerical_gradient(x, t)
print(grads["W1"].shape,
      grads["b1"].shape,
      grads["W2"].shape,
      grads["b2"],
      sep="\n")

# 가중치 초기값은 다 0으로 초기화하면 경사하강법을 제대로 움직이지 못하게 하고,
# 특정 수로 주자니 반복수 내에 값이 수렴하지 않을 수도 있음. 등등을 이유로 임의의 값으로 초기화한다.
# 는 이걸 써야할지 말아야 할지.

# 4.5.2 미니배치 학습 구현하기
