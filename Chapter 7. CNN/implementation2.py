# Chapter 7. 합성곱 신경망 (CNN)
# * 7.1 전체 구조
# 기존: (완전 연결) Affine -> Activation
# Convolution -> Activation -> Pooling

# 각 계층 사이에 입체적인 데이터가 흐른다는 점에서 완전연결 신경망과 구분됨.
# 완전 연결 신경망은 데이터의 위치 정보를 무시했음.
# 그러나 CNN은 이미지 데이터를 기준으로 하기 때문에 픽셀의 위치도 중요한 정보로 생각함.


# * 7.2 합성곱 계층
# 7.2.2 합성곱 연산 (필터 연산)
# mask(or filter, kernel)이 움직이면서 대응되는 윈도우와의 계산
# 차원이 축소되는 or 외곽 데이터의 가중치 낮은 가중치 비율 weakness -> padding을 통해 해결
# 원한다면 padding을 통해 차원을 더 늘릴 수도 있음.
# 또는 mask가 움직이는 단위 stride를 키워서 차원을 축소시킬 수도 있음.
# OH = (H + 2P - FH) / S + 1
# OW = (W + 2P - FW) / S + 1

# SciPy는 합성곱과 교차상관을 플리핑이라는 기준으로 확실하게 분리하지만, 딥러닝은 그렇지 않다고.
# 합성곱 연산을 가중치를 통한 곱연산으로 취급하고, 편향을 더하면 이전과 비슷한 형태를 구성할 수 있음.

# 7.2.5 3차원 데이터의 합성곱 연산
# input(H, W) (*) filter(FH, FW) + bias(1) => output(OH, OW)

# input(C, H, W) (*) filter(C, FH, FW) + bias(1) => output(C, OH, OW)
# 하나의 채널에서 (보통 3개) 다량의 채널로

# input(C, H, W) (*) filter(FN, C, FH, FW) + bias(FN, 1) => output(FN, OH, OW)
# 하나의 필터에서 여러 개의 필터로 (하나의 노드에서 여러 노드로)

# input(N, C, H, W) (*) filter(FN, C, FH, FW) + bias(FN, 1) => output(FN, OH, OW)
# 한 번에 하나의 연산에서 sampling 연산으로
# 모든 것을 한 번만 이용해서 learn -> 적은 경우의 수를 제대로 반영 X
# 하나 계산해서 반영하고, 하나 계산해서 learn -> 너무 오래 걸림, overfitting 위험성 존재.
# 그러니까 sampling 값으로 모형을 학습하자 (Loss 관점)


# * 7.3 풀링 계층
# kernel을 움직이는데, stride는 윈도우가 겹치지 않는 값으로 결정된다. (즉, kernel이 3X3의 크기라면 stride도 3이라는 뜻.)
# 윈도우에서 대푯값을 추출하므로 그냥 차원을 줄이는 거 (최대 풀링, 평균 풀링)
# 가중치가 필요 없다, 채널 수가 변하지 않는다, 입력의 변화 영향을 적게 받는다
# 추가적으로 해상도가 높다면 잡음을 줄일 수 있음.


# * 7.4 합성곱/풀링 계층 구현하기
# 7.4.1 4차원 계열

import numpy as np

x = np.random.rand(10, 1, 28, 28)  # shape을 정해주면 무작위로 생성
# print(x.shape, x[0].shape, x[0][0], x[0, 0])

# 7.4.2 im2col로 데이터 전개하기
# 4차원은 반복문을 많이 써서 numpy 패키지에 overhead를 줌. (정확히는 성능 떨어짐.)
# 그래서 overhead를 최대한 줄이는 방향으로, 다시 변형을 통해 차원을 낮춰보자.
# 행렬 곱셉에 특화된 선형 대수 관련 계산 라이브러리를 이용

# im2col -> (from) image to column / col2im: (from) column to image
# (N * (image에 만들어질 수 있는 window 수), C * H * W)

# im2col(input) %*% (filter matrix)^T => (M X N) => col2im(output)
# 일반적으로 Xi의 인스턴스 및 확장을 살펴보면 위에 식에 filter는 행이 (새로운) 위치를 의미하므로 transpose된 형태로 보는 게 맞는 듯하다.

# 7.4.3 합성곱 계층 구현하기
# im2col(input_data, filter_h, filter_w, stride=1, pad=0)

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from mnist_test.utils import im2col

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col1.shape, col2.shape)
# (N * (image에 만들어질 수 있는 window 수), C * H * W)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
        out_w = int((W + 2 * self.pad - FW) / self.stride + 1)

        col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)
        col_w = self.W.reshape(FN, -1).T
        out = np.dot(col, col_w) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # reshape는 퍽 친절하다, -1을 건네주므로써 resize와 같은 효과를 얻는다. (데이터를 잃어버리지 않는다)
        # 그리고 transpose를 통해 input과 유사한 구성으로 재정렬한다.

    # backward는 col2im를 이용하는데, reshape 이후가 장난이 아닌데


# 7.4.4 풀링 계층 구현하기
# pooling의 im2col는 채널마다 윈도우마다 다른 데이터처럼 취급하는 것처럼 보임.


class Pooling:
    def __init__(self, pool_h, pool_W, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_W
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int((H - 2 * pool_h) / self.stride + 1)
        out_w = int((W - 2 * pool_w) / self.stride + 1)

        # 1. 입력 데이터를 전개한다.
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 2. 행별 최댓값을 구한다.
        out = np.max(col, axis=1)
        # 3. 적절한 모양을 성형한다.
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        # 왜 이런 식으로 정렬할 수밖에 없냐고 묻는다면, im2col 구성을 확인하라는 말밖에 못해줌.
        return out

        # backward는 ReLU의 max 역전파를 참고하라는데, 그러면 mask를 만들어 저장해야 하는 건가?


# * 7.5 CNN 구현하기
# SimpleConvNet: Conv. -> ReLU -> Pooling -> Affine -> ReLU -> Affine -> Softmax


class SimpleConvNet:
    # 설마 중간 매개변수에 대한 default를 설정할 수 없다니 Python은 C언어로 구성했다는 걸 알 수 있는 대목이었다.
    # 정확히는 뒤에 매개변수도 default를 설정해달라고 했다.
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
        conv_output_size = (input_size + 2 * filter_pad - filter_size) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        # 사실 왜 2로 나눈 걸 쓰는 건지 잘 모르겠다, 최대 크기를 잡는 것가?

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers = OrderDict()
        # 1층: CNN
        self.layers["Conv1"] = Convolution(self.params["W1"], self.param["b1"], conv_params["stride"], conv_params["pad"])
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 2층: 완전 연결
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        # 3층: 출력에 연결된 층
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db

        return grads

# CNN: 이미지 데이터가 가진 지리적 특성을 이용하라

# * 7.6 CNN 시각화하기
