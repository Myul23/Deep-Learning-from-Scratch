# Chapter 6. 학습 관련 기술들

# 가중치 매개변수의 최적값을 위한 최적화 방법, 가중치 매개변수 초깃값, hyper-parameter 설정 방법
# Overfitting 막기: 가중치 감소, drop-out
# 신식 문물: 배치 정규화

# 6.1 매개변수 갱신
# 모델 최적화: loss의 최소화하는 매개변수 -> 매개변수 최적값
# 현재(배운 데)까지는 넓은 매개변수 공간에서 SGD라는 단순한 방법으로 매개변수를 최적화하고자 했음.

# 6.1.1 모험가 이야기
# 이성건 교수님께 들은 눈가리고 낮은 곳 찾기를 모험가 이야기로 풀어 설명
# 지금 서 있는 장소에서 가장 크게 기울어진 방향으로 가자는 것이 SGD의 전략

# 6.1.2 SGD (Stocastic Gradient Descent)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
        # 아이고 내가 어제 설명에 dictionary에 대한 설명을 이상하게 했구나.


# 6.1.3 SGD의 단점
# 지역 최솟값 구간에 걸리지 않는다면, SGD에 의한 최적화는 잘 이루어지겠지만, 많이 반복해야 함.
# 이를 그림으로 표현하면 (loss contour가 타원을 그릴 때) 지그재그( 모양새)를 그리며 최적화됨.
# 비효율적인 움직임 -> 비등방성(anisotropy)함수에서는 탐색 경로가 비효율적이다.
# 게다가 기울어진 방향(손실 함수 미분 방향)이 전체 최솟값과 다른 방향을 가리켜서라는 점도 생각해야 한다.
# - loss contour가 굉장히 찌그러진 타원인 경우, 기울기 방향이 찌그러진 축의 수직(?)이 되어 그쪽으로 많이 움직임.
# 따라서 이때는 무작정 기울어진 곳으로 움직이는 함수보다 더 똑똑한 함수를 이용하자.
# 새로운 Gradient Descent 방법으로 모멘텀, AdaGrad, Adam을 소개하고자 함.

# 6.1.4 모멘텀 (Momentum): 운동량
# 전에 배울 땐 local optimum에서 빠져나오고자 고안된 가속도의 추가 정도로 배웠음.
# v := av - (에타)(loss에 대한 W의 미분값), W := W + v (a는 0.9 정도?)
# 매개변수에 일정 비율만큼 움직이는 것이 아니라 현재 손실함수의 기울기에 대한 일정 비율만큼 매개변수를 움직이자.
# av: 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할, 지면 마찰이나 공기 저항에 해당


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# 모멘텀은 기존의 Gradient Descent보다 덜 지그재그 거리며 최적화 값을 찾는다.

# Nesterov에 대한 클래스가 있음.

# 6.1.5 AdaGrad
# 전에 배우기론 학습률의 크기를 제어 못하면 최적값을 찾지 못하고 그 주위를 맴돌다가 끝남.
# 이를 줄이고자 일정 단계가 지나면 학습률을 (더 작게) 바꾸자는 아이디어가 나옴. -> 학습률 감소 (learning rate decay)
# AdaGrad는 각각의 매개변수에 맞춤형 값을 만들어줍니다. -> 각각의 매개변수 상황에 맞게 학습률 감소를 진행합니다.
# h := h + (dL/dW) (원소간 곱) (dL/dW), W := W - (에타) * (1/sqrt(h)) * (dL/dW)
# 여기서 중요한 것은 현재 기울기의 영향이 이전 기울기의 영향만큼(?) 덜 움직인다는 것.
# 요약하자면 많이 움직인 원소는 학습률이 낮아진다는 뜻.

# 그러나 AdaGrad는 이전 기울기 값이 쌓이면서 순간 갱신량이 0이 되어 전혀 갱신하지 않는 지점이 생김.
# 이 문제를 개선한 방법이 RMSProp (지수이동평균, 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영)
# exp(y_t-1) + yt? 저번 학기에 배웠는데 기억이 안 난다...


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            # self.h[key]에 어떤 위치에 0이 있을 수도 있어서
            # 대부분의 딥러닝 프레임워크에서는 이 값도 인수로 설정할 수 있음.


# loss contour를 통한 움직임을 보면 마치 지그재그로 움직일 것을 예상하고 다음 움직임은 조금만 움직여라고 명령한 듯 보인다.
# 처음부터 효율적으로 움직이는 Gradient Descent에는 별로일 것 같은 방법이지만,
# 다양한 loss contour를 가지는 분포에 대해 범용적( 또는 평균적)으로 (대체로) 효율적으로 적용할 수 있다는 점에서 일단 찬사를 보낸다.

# RMSprop에 대한 클래스가 구현되어 있음.

# 6.1.6 Adam (2015)
# Momentum과 AdaGrad의 융합 + hyper-parameter의 편향 보정도 진행
# 앞서 고민스러웠던 부분을 해결, 근데 원논문 참고하라면서 원논문이 뭔지 안 알려줌.
# https://arxiv.org/abs/1412.6980v8

# 6.1.7 어느 갱신 방법을 이용할 것인가?
# 찌그러진 타원에 경우엔 AdaGrad가 가장 좋은 방향을 가졌음.
# 그러나 손실 함수는 형태의 따라, 즉 데이터에 따라 형태가 달라지고 특정 형태에 최적인 GD가 있는 것뿐.
# 뭐가 가히 범접할 수 없을 정도의 영역이다라고 말할 수 없음.
# 따라서 SGD, 모멘텀, AdaGrad, Adam는 각각 잘 푸는 문제와 서툰 문제가 있는 것임.

# 결과적으로 기본으로는 Adam을 이용하는데, non-stationary가 심한 문제(강화학습 등)에서는 RMSprop를 선택하는 경우도 많다.

# 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교
# 확실히 SGD가 그냥 느리기도 하고, 천천히 가는구만.

# 6.2 가중치의 초깃값
# 6.2.1 초깃값을 0으로 하면?
# Overfitting을 줄이고자 가중치를 작게 만든다 -> 작게 시작하자.
# 초깃값을 0 -> 순전파 때 다음 뉴런에 모두 같은 값(b)을 전달
# -> 역전파 때 다음 층 가중치가 받는 미분값이 입력이 b로 동일하므로 b로 같은 값을 곱하게 됨.
# -> 숫자도 같게 움직임. -> 오차역전파법에서 모든 가중치의 값이 똑같이 갱신됨.
# 그래서 가중치의 대칭적인 구조를 무너뜨려야 함. transpose를 하면 다른 값이 곱해질 수 있도록

# 6.2.2 은닉층의 활성화값 분포
# 은닉층의 활성화값의 분포를 관찰하면 중요한 정보를 얻을 수 있음?
# Visual ... recognition: http://karpathy.github.io/neuralnets/

import numpy as np
import matplotlib.pyplot as plt


# 이미 정의한 걸 불러와 쓸 수 있지만, 그냥 정의하는 걸로
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(1000, 100)  # ~ N(0, 1)에서 (1000, 100) 행렬 뽑기
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 1. w = np.random.randn(node_num, node_num) * 1
    # 2. w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    # 여기서는 가중치가 같은 대칭 모양이라 이렇게 썼지만, 다음부터는 이전 node 갯수를 저장해야 함.
    # ReLU: w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num) & ReLU
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

# 맞다, sigmoid의 중간값은 0.5(x = 0 -> y = 0.5)지.
# activation function이 sigmoid이고 값이 0과 1에 몰리면 업데이트가 조금 되므로 전달되는 역전파 값이 0에 가까워짐.
# 이게 층을 거듭하면서 값이 작아지고 작아지다가 사라져버림. 이를 기울기 소실 (gradient vanishing)이라 함.

# 표준편차를 0.01로 하면 기울기 소실은 사라지지만, 다수의 뉴런이 거의 같은 값을 출력해서 뉴련을 여거 개 둔 의미가 없다진다.
# 이를 뉴런들의 표현력을 제한한다는 관점에서 문제가 생긴다고 말한다.

# Xavier Glorot & Yoshua Bengio (사비에르 글로트 & 요슈아 벤지오)
# Xavier 초깃값: 표준편차를 1/sqrt(입력의 갯수)로 하자.
# Caffe 프레임워크는 가중치 초깃값을 설정할 때 인수로 xavier를 지정 가능, tensorflow는 선언해서 넘겨야 함.
# 활성화함수를 tanh를 쓰면 층을 거듭해도 종모양을 유지하지만, 활성화 함수용으로는 원점에서 대칭인 함수가 바람직해 sigmoid를 더 사용한다고.
# Xavier 초깃값은 활성화 함수가 선형인 것을 전제로 했음. 그래서 sigmoid, tanh 등에 적합

# 6.2.3 ReLU를 사용할 때의 가중치 초깃값
# ReLU는 입력 범위의 절반이 0이 되므로 부여할 가중치 공간을 2배로 늘려줬음. 이를 찾아낸 사람의 이름을 따 He 초깃값이라 함.
# He 초깃값: (가중치 초깃값) 표준편차를 sqrt(2 / 입력의 갯수)로 하자.
# 신경망에 아주 작은 데이터가 흐른다는 것은 역전파 때 가중치의 기울기 역시 작아진다는 뜻 -> 실제로 학습이 거의 이뤄지지 않을 것.
# ReLU -> He 초깃값, Sigmoid, tanh etc. -> Xavire

# 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교
# weight_init_compare.py

# 6.3 배치 정규화
# 가중치 초깃값을 골고루 설정해야 함을 배웠음. 그렇다면 그 값을 강제해보면 어떨까.
# 이런 아이디어로 배치 정규화(Batch Normalization)가 출발했음.

# 6.3.1 (2015)
# 배치 정규화는 학습을 빨리 진행할 수 있고, 초깃값에 크게 의존하지 않으며, 오버피팅을 억제함.
# 따라서 배치 정규화를 통해 몇 가지 골치아픈 문제를 상관하지 않을 수 있음.

# 배치 정규화 계층의 삽입
# 마치 활성화 함수로 인해 기울기 값이 제대로 연산되지 않는 것을 최소화하는 것처럼 보임.
# 정규분포에서의 평균, 분산 구하기 그리고 그것들을 이용한 정규화(평균 0점 맞추기, 분산 안정화?)
# 얘는 값을 갱신하는 거라 0이 될 수도 있음. 그래서 나눌 때 표준편차가 0이 되는 것을 방지하고자 엡실론 더함.
# 아.. 활성화 함수 뒤에도 넣을 수 있구나.
# 정규화를 식으로 표현하면 y = rx + b의 형태가 될 것임. r의 초깃값은 1, b의 초깃값은 0
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

# 6.3.2 배치 정규화의 효과
# 배치 정규화를 이용한 것이 실선으로 초기 가중치에 엄청난 영향을 받지 않는다.
# 반면에 배치를 이용하지 않은 모형에 대해선 특정 가중치에선 아예 정확도가 오르지 않기도 했다.
# 학습이 빨라지며 가중치 초깃값에 크게 의존하지 않음.
# test_mnist.py

# 6.4 바른 학습을 위해
# 과대적합이 되는 일이 비일비재

# 6.4.1 오버피팅
# 대체로 매개변수가 많고 표현력이 높은 모델(??), 훈련 데이터가 적음 -> 오버피팅으로 이어지는 경우가 많음
# test_mnist_overfitting.py
# 에폭 단위(모든 훈련 데이터를 한 번씩 본 단위)의 정확도를 저장합니다

# 6.4.2 가중치 감소
# 큰 가중치에 대해 큰 패널티를 부과하는 방식
# 손실함수에 대한 패널티 항은 모든 가중치의 제곱합으로 구성됨.
# 미분했을 때에 보다 원형에 가까운 값을 주고자? 내려오는 지수 값을 막을 1/2가 곱해져 있음.
# 쨌든 결과적으로 (람다)W를 패널티 기울기로 전달함.
# 이걸 그래프로 보면 아직까지 차이는 있지만, 전보단 줄었음을 알 수 있음.
# train의 폭발적인 과대적합(100% 도달)은 막은 것으로 보임.

# 6.4.3 드롭아웃
# 신경망 모델이 더욱 복잡해지면 가중치 감소만으론 과대적합을 막기 어려워짐.
# 그래서 dropout을 이용함.
# dropout: 훈련 때 임의로 몇 개의 노드만을 가지고 학습시키는 것.
# 그리고 이후 예측에선 전체 노드를 이용, 단 각 뉴런의 출력에 훈련 때 삭제 안 한 비율을 곱하여 출력
# 실제 딥러닝 프레임워크들도 삭제 안 한 비율은 잘 곱하지 않음. 왠지 tensorflow 가르쳐주실 때 없더라니.


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        # 학습마다 임의로 구성된 mask를 통해 특정 노드를 사용하고 사용하지 않고 결정

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 원래 shape의 반환은 리스트니까 그 안에 값만 가져오도록
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


# 결과적으로 과대적합을 줄이는 건 우리가 어떻게 할 수 없는 test error를 낮추는 방법이긴 하지만,
# 실제론 그게 아니라 좀 더 효율적인 training, 조금 더 높은 training error를 구하게 한 것과 같다.

# 6.5 적절한 하이퍼파라미터 값 찾기
# 신경망에서 hyper-parameter란 학습에 관여하는 (모수)변수들을 의미
# 매우 중요하지만, 최적의 값을 찾기란 쉽지 않습니다.

# 6.5.1 검증 데이터
# 지금까지 총 가진 데이터 = 훈련 데이터 + test(시험) 데이터였는데
# hyper-parameter 조정에 쓸 세번째 분류를 만듦. 총 데이터 = 훈련 데이터(초기 훈련 + 모수 검증) + test
# test 데이터로 hyper-parameter의 값을 조정하면 결국 학습시킨 게 되어버림 -> Overfitting

# train(train(훈련) + validation(검증)) + test(시험)
# validation set을 얻는 가장 쉬운 방법은 train에서 20%를 validation으로 (분리해) 쓰는 것.
# test_mnist_hyper_parameter.py

# 6.5.2 하이퍼파라미터 최적화
# hyper-parameter의 최적 값이 존재하는 범위를 조금씩 줄여간다는 것이 핵심
# 최종 정확도에 미치는 영향이 hyper-parameter마다 다르기 때문에 최적의 hyper-parameter 쌍이 있을 공간으로 줄여나가는 게 더 좋은 결과를 준다고.

# hyper-parameter의 범위를 대략적으로 지정하는 것이 효과적
# (0.001, 1000)과 같이 10의 거듭제곱 단위로 범위를 지정 : log scale로 지정한다고 함.
# 딥러닝은 학습에 (생각보다 더) 오랜 시간이 걸리기 때문에 나쁠 듯한 값은 일찍 포기하는 것이 좋다고?
# 에폭을 작게 -> epoch으로 평가할 만한 크기를 작게 하자. (다른 것들과 같은 비율을 고집한다면 batch_size가 커지는 게 맞긴 함.)

# 따라서 hyper-parameter를 갱신하는 과정은 대략적으로 값의 범위를 설정하고
# -> 범위 안에서 무작위로 추출하고 -> 학습해서 검증 데이터로 정확도를 평가 (에폭 작게)
# -> 특정 횟수 반복하며 정확도를 통해 hyper-parameter 범위를 좁힘. -> 을 반복하다가 하나를 select

# addition) 현재는 수행자의 지혜과 직관에 의존한다는 느낌이 들 수 있다.
# 더 세련된 기법으로는 베이즈 정리를 기반으로 하는 Bayesian Optimization이 있다.
# <논문 Practical Bayesian Optimization of Machine Learning Algorithms>

# 6.5.3 하이퍼파라미터 최적화 구현하기
# 앞서 소개한 것처럼 임의의 구간 지정 -> 무작위 추출
# test_mnist_hyper_parameter.py
