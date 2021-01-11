# Chpater 5. 오차역전법

# 5.1 계산 그래프
# 5.1.1 계산 그래프로 풀다
# 순전파 (forward propagation): 계산 그래프의 출발점부터 종착점으로의 전파
# 역전파 (backward propagation): 계산 그래프의 종착점부터 출발점으로의 (가중치 계산) 전파

# 5.1.2 국소적 계산
# 각 노드에서의 계산은 국소적
# 각 노드는 해당 값이 어떻게 구성되었는지 확인할 필요 없음
# 국소적 계산은 단순하지만, 그 결과를 전달함으로써 전체를 구성하는 복잡한 계산을 해낼 수 있음.

# 5.1.3 왜 계산 그래프로 푸는가
# 전체가 아무리 복잡해도 각 노드에서는 단순한 계산에 집중하여 문제를 단순화할 수 있음.
# 중간 계산 결과를 모두 보관할 수 있음.

# 가장 큰 이유는 역전파를 통해 미분을 효율적으로 계산할 수 있다는 점에 있음.
# 각각의 미분을 통해 출력에 대한 입력의 변화 정도를 구할 수 있고,
# 출력에 얼마만큼의 영향을 주는지를 가중치를 설정할 수 있다.

# 5.2 연쇄법칙
# 5.2.2 연쇄법칙이란
# 연쇄법칙: 합성함수의 미분이 합성함수를 구성하는 각 함수의 미분의 곱으로 이루어진다.

# 5.3 역전파
# 5.3.1 덧셈 노드의 역전파
# 덧셈 연산만 봤을 때 덧셈 노드의 역전파는 입력값(상류로부터 온 역전파 값)을 그래도 흘려보낸다.

# 5.3.2 곱셈 노드의 역전파
# 곱셈 연산만 봤을 때 곱셈 노드의 역전파는 입력값에 다른 입력 신호를 곱해서 흘려보낸다.
# 곱셈의 역전파는 순방향 입력 신호의 값을 다른 입력 신호로 사용합니다.

# 5.4 단순한 계층 구현하기
# 계층: 신경망의 기능 단위 (시그모이드 함수를 위한 Sigmoid, 행렬 곱을 위한 Affine 기능)


# 5.4.1 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 순방향 값 유지를 위한 공간 만들기

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# 순전파 확인
apple = 100
apple_num = 2
tax = 1.1
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# 역전파 확인
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)


# 5.4.2 덧셈 계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# apple = 100
# apple_num = 2
orange = 150
orange_num = 3
# tax = 1.1

# 계층
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 다시 순전파 확인
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
total_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(total_price, tax)

# 다시 역전파 확인
# dprice = 1
dtotal_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dtotal_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_num, dapple, dorange_num, dorange, dtax)

# 5.5 활성화 함수 계층 구현하기
# 5.5.1 ReLU 계층
# ReLU: 0 초과는 입력 값 그대로 -> 출력 = 입력
#       0 이하는 0 -> 출력 = 0


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


import numpy as np
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
mask = (x <= 0)
print(x, mask, sep="\n")

# 5.5.2 Sigmoid 계층
# x -> (* -1) -> (exp) -> (+ 1) -> (^ -1)
# 뒤에서부터 차례대로 미분하기 (p167 - p169)


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


# 5.6 Affine/Softmax 계층 구현하기
# 5.6.1 Affine 계층
# Affine 변환: 순전파 때 수행하는 행렬의 곱을 기하학에서 부르는 말
# 여기서부터는 행렬이 흐른다, 행렬 미분을 설명해 말아? 대충은 할 수 있을 것도 같은데

# 5.6.2 배치용 Affine 계층
# 가중치와 마찬가지로 편향도 N개의 데이터가 되면서 기존의 b의 shape을 유지하면서
# 연산된 각 위치를 역전파 값에 더함.
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
print(X_dot_W, X_dot_W + B, sep="\n")


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


# 5.6.3 Softmax-with-Loss 계층 (p294 - p298)
# 결과적으로 Cross-Entropy-Error 계층과 Softamx 계층은 예측과 실제 사이의 차이를 역전파 값으로 전달한다.
# 더불어 항등함수의 손실함수로 오차제곱합을 사용하면 Softmax & Cross-Entropy-Error 혼합 계층과 같이 예측과 실제값의 차이를 역전파 값으로 전달한다.


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# 5.7 오차역전파법 구현하기
# 5.7.1 신경망 학습의 전체 그림
# 4장에서 미니배치 -> 기울기 산출 -> 매개변수 갱신 -> 반복

# 5.7.2 오차역전파법을 적용한 신경망 구현하기 (코딩)

# 5.7.3 오차역전파법으로 구한 기울기 검증하기 (코딩)
# 기울기 확인: 해석적으로 구한 미분(기울기)을 수치 미분을 통해 확인 및 검증하는 것.

# 5.7.4 오차역전파법을 사용한 학습 구현하기 (코딩)
