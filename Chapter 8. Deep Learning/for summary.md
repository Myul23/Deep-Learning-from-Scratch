## 딥러닝 (Deep Learning)

딥러닝은 층을 깊게 한 심층 신경망

### 8.1 더 깊게

#### 8.1.1 더 깊은 신경망으로

<!-- VGG 신경망을 참고 -->

input -> 3 * (Conv -> ReLU -> Conv -> ReLU -> Pool) -> Affine -> ReLU -> Dropout -> Affine -> Dropout -> Softmax

- filter: (3, 3)
- Activation: ReLU, Weight initialization: He-초깃값
- 'Affine -> Dropout' 형태로 Regularization을 이용
- Optimizer: Adam

[What is the class of this image?](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)

데이터 확장에 주목해보자.
데이터 확장이란 입력 이미지를 회전하거나 세로로 이동하는 등 미세한 변화를 주어 이미지의 개수를 늘리는 것
- 데이터가 적을 때 효과적
- crop: 이미지 일부를 잘라내는 기법
- flip: 대칭성을 고려하지 않아도 되는 경우에 좌우를 뒤집는 기법
- 밝기 변화, 외형 변화, 스케일 변화 (확대, 축소) 등

#### 8.3.1 깊게 하는 이유 (feat. ILSVRC)

```
      / (7, 7)                   : 49 \
input - (5, 5) - (3, 3)          : 34 => (input's 7, 7)
      \ (3, 3) - (3, 3) - (3, 3) : 27 /
```

- 층을 깊게 한 신경망은 깊지 않은 경우보다 적은 매개변수로 같은 (혹은 그 이상) 수준의 표현력을 달성할 수 있다.
- 매개변수의 수를 줄여 넓은 수용 영역(receptive field)를 소화할 수 있다.
  - 수용 영역: 뉴런에 변화를 일으키는 국소적인 공간 영역
- 층을 거듭하면서 활성화 함수를 합성곱 계층 사이에 끼움으로써 신경망의 표현력이 더욱 비선형적으로 개선됨.
- 또 당연하게 학습할 매개변수가 줄면서 학습 속도가 빨라짐.
- CNN의 관점에서 봤을 때, 층을 깊게 할수록 더 복잡하지만 추상적인 고급 정보를 캐치할 수 있음.

### 8.2 딥러닝의 초기 역사 

ILSVRC (ImageNet Large Scale Visual Recognition Challenge)

2012: AlexNet

#### 8.2.1 이미지넷

ImageNet: 100만 장이 넘는 이미지를 담고 있는 데이터셋

#### 8.2.2 VGG

ILSVRC: 2014, 2

input (3, 224, 224) - Conv (?, 224, 224) - Pool (?, 112, 112) - Conv (?, 112, 112) - Pool (?, 56, 56) - Conv (?, 56, 56) - Conv (?, 56, 56) - Pool (?, 28, 28) - Conv (?, 28, 28) - Conv (?, 28, 28) - Pool (?, 14, 14) - Conv (?, 14, 14) - Conv (?, 14, 14) - Pool (?, 7) - Flatten (4096) - Dense (4096) - Dense (1000)

#### 8.2.3 GoogLeNet

인셉션 구조: input - 4개로 분할 (Conv(1), Conv(3), Conv(5), Pool(3))

#### 8.2.4 ResNet (Residual Network)

- 층이 깊어질수록 가중치 소실 및 역전파로 인한 전체적인 accuracy 감소 문제를 해결하고자
- 그러나 여전히 층을 깊게 하는 데 한계가 있는 건 사실

스킵 연결 (skip connection)
- 입력 데이터를 그대로 흘리는 것
- 스킵 연결로 기울기가 작아지거나 지나치게 커질 걱정 없이 앞 층에 의미 있는 기울기가 전해지리라 기대할 수 있다.

```
input - Conv - ReLU - Conv - + - ReLU
      \-----(identity)------/
```

- 전이 학습 (transfer learning): 특정 데이터에 대해서 학습된 가중치(혹은 그 일부)를 다른 신경망에 복사한 다음, 그 상태로 재학습을 수행하는 방법
  - 학습용 데이터셋이 적을 때 특히 유용한 방법
  - 구성이 같은 신경망을 준비하고, 미리 학습된 가중치를 초깃값으로 설정한 후, 새로운 데이터셋을 대상으로 재학습(fine tuning)을 수행하는 것

### 8.3 더 빠르게(딥러닝 고속화)

최근 프레임워크에선 학습을 복수의 GPU(Graphics Processing Unit)와 여러 기기로 분산 수행하기 시작했다. (Hadoop)

#### 8.3.1 풀어야 할 숙제

Convolutional 계층처럼 연산이 많은 계층이 당연히 오래 걸린다.<br />
그래서 이렇게 연산이 많은 계층을 어떻게 하면 더 짧은 시간 안에 효율적으로 계산할 수 있을까가 핵심

#### 8.3.2 GPU를 활용한 고속화

GPU는 원래 그래픽 전용 보드에 이용했음. 또는 아주 기본적인 단순 계산에만 이용해왔음.

GPU 컴퓨팅: GPU를 단순 범용 연산에 이용하는 것. (GPU는 병렬 수치 연산을 고속으로 처리할 수 있어 이득)

아직까지 GPU 컴퓨팅은 엔비디아의 GPU 컴퓨팅용 통합 개발 환경인 CUDA를 이용하는 식으로 되어 있어서 AMD로는 CUDA 위에서 동작하는 라이브러리, cuDNN을 이용할 수 없다.

참고: tensorflwo GPU도 이용할 수 없다.

#### 8.3.3 분산 학습

Google's Tensorflow, Microsoft's CNTK (Computational Network Toolkit)이 분산 학습에 역점을 두고 개발되고 있음.

거대한 데이터센터의 저지연(low latency), 고처리량(high throughput) 네트워크 위에서 이 프레임워크들이 수행하는 분산 학습은 놀라운 효과를 보이고 있습니다.

GPU를 100개까지 활용하면 하나일 때보다 56배 빨라짐을 알 수 있었고, 이는 7일짜리 작업이 부로가 3시간 만에 끝난다는 결과를 내기도 함.

계산을 어떻게 분산시키느냐는 컴퓨터 사이의 통신과 데이터 동기화 등을 다 고려해야 하는 몹시 어려운 문제.

#### 8.3.4 연산 정밀도와 비트 줄이기

계산 능력 외에도 메모리 용량과 버스 대역폭 등이 딥러닝 고속화에 병목이 될 수 있다.

- 메모리 용량 면에서 대량의 가중치와 중간 데이터를 메모리에 저장해야 한다는 것을 생각해야 함.
- 버스 대역폭 면에서 GPU(혹은 CPU)의 버스를 흐르는 데이터가 많아져 한계를 넘어서면 병목

따라서 통신에 이용되는 데이터를 최소로 

다행히 딥러닝은 입력 이미지에 노이즈가 섞여도 출력 결과가 크게 달라지지 않습니다. 결과적으로 데이터를 저장하고 전달할 때 수치 정밀도(완벽하게 이전과 같은 데이터여야 하는 정도)가 높지 않아도 됩니다. (이런 견고성 덕분에 신경망을 흐르는 데이터를 퇴하시켜도 출력에 주는 영향은 적다)

컴퓨터에서 실수를 표현하는 방식: 32비트 단정밀도(single-precision), 64비트 배정밀도(double-precision) 부동소수점 등이 있음. 딥러닝은 16비트 반정밀도(half-precision)만 사용해도 학습에 문제가 없다고 알려져 있음.

실제로 Pascal (2016, NVDIA's GPU architect)
엔비디아의 Maxwell 세대 GPU는 반정밀도 부동소수점 수를 storage(데이터를 저장하는 기능)로 지원하고 있었지만, 연산 자체는 16비트로 수해앟지 않았음. 파스칼 세대에 와서 연산 역시 16비트로 진행하며 2배 정도 빨라짐.

AMD GPU 역시 VEGA 모델부터는 반정밀도 연산을 지원

Python은 64비트 배정밀도 부동소수점을 NumPy는 16비트 반정밀도 부동소수점으로 표현(show or print)

Binarized Neural Networks: (최근) 가중치와 중간 데이터를 1트로 표현

딥러닝을 고속화하기 위해 비트를 줄이는 기술은 앞으로 주시해야 할 분야이며, 특히 딥러닝을 임베디드용으로 이용할 때 중요한 주제입니다.

### 8.4 딥러닝의 활용

좀 더 보편적인 사물 인식의 분야로

#### 8.4.1 사물 검출

R-CNN (Regions with Convolutional Neural Network): CNN을 이용한 사물 검출
- 입력 이미지 -> 후보 영역 추출 -> CNN 특징 계산 -> 영역 분류
- 후보 영역 추출은 적당히 잡아줘야 함. (Selective Search 기법 이용)

Faster R-CNN
- 후보 영역 추출까지 CNN으로 처리하는 방법
- 딥러닝 end-to-end 성질에 더 가까운 기법임.

R-CNN과 Faster R-CNN 하는 게 마치 초기 Perceptron과 Neural Network간 관계를 보는 듯하다.

#### 8.4.2 분할 (segmentation)

가장 단순한 분할 방법은 모든 픽셀에 class를 붙이는 것.<br />
-> 픽셀의 수만큼 forward 처리를 해야 하여 긴 시간이 걸림. (합성곱 연산에서 많은 영역을 쓸데없이 다시 계산하게 됨)

FCN (Fully Convolutional Network): 한 번의 forward 처리로 모든 픽셀의 클래스를 분류해줌.
- 일반적인 CNN이 출력에 가까운 계층에선 FC를 이용하는 것과 달리 비슷한 역할을 하는 Convolutional 계층으로 대체하여 마지막 출력까지 공간 볼륨을 유지할 수 있음.
- 마지막에 원래 공간 크기대로 공간을 확대하는 처리를 도입. -> 이중 선형 보간(bilinear interpolation) 기법을 역합성곱(deconvolution) 연산으로 구현

#### 8.4.3 사진 캡션 생성

NIC (Neural Image Caption): 이미지를 보고 문장 생성
- CNN + RNN

Multimodal Processing (멀티모달 처리): 사진이나 자연어와 같은 여러 종류의 정보를 조합하고 처리하는 것 (분야)

### 8.5 딥러닝의 미래

#### 8.5.1 이미지 스타일(화풍) 변환

A Neural Algorithm of Artistic Style

네트워크의 중간 데이터가 콘텐츠(그리고자 하는) 이미지의 중간 데이터와 비슷해지도록 학습<br />
- 입력 이미지를 콘텐츠 이미지의 형태로 흉내낼 수 있음.
- 스타일 행렬 개념의 도입 (이 스타일 행렬의 오차를 줄이도록 학습하여 콘텐츠 이미지를 스타일 이미지의 화풍으로 변환시킴)

#### 8.5.2 이미지 생성

DCGAN (Deep Convolutional Generative Adversarial Network)
- 대량의 학습 데이터를 통해 학습하여 반대로 클래스에 속하는 이미지를 만들어내도록 할 수 있음.
- 이름을 보면 알 수 있지만, Deep-GAN 방식
  - 생성자(Generator), 식별자(discriminator)로 불리는 2개의 신경망을 이용
  - 생성자가 진짜와 같은 이미지를 생성하고 식별자는 그것이 생서앚가 생성한 것인지 아니면 실제로 촬영된 이미지인지 판별.
  - 이 둘이 학습을 통해 겨루면서 더 정교한 가짜 이미지를 만들어내게 됨. (이것이 GAN)
  - Deep Belief Network, Deep Boltzmann Machine과 같은 Unsupervised Learning

#### 8.5.3 자율 주행

SegNet: CNN 기반 신경망으로 주변 환경을 다양한 클래스로 분할 및 인식

#### 8.5.4 Deep Q-Network

강화학습 (Reinforcement Learning)
- 에이전트는 예상되는 보상을 받기 위해 환경을 바꾸고, 바뀐 환경에 따라 (실제적인) 보상을 받고.

DQN 연구에는 비디오 게임 영상을 입력받아 플레이하게 하는 연구가 있다.
