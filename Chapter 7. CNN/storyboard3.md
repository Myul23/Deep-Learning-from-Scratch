7장의 5절 CNN의 구현부터의 내용을 다룹니다.

CNN 구현에 들어가기에 앞서 / Fully Connected과 Convolution 계층에 대한 / 차이를 보고자 합니다.

완전 연결 계층은 / 보이는 것처럼 계산의 단위가 / input의 행과 weight의 열입니다. / 

이와 다르게 Convolution 계층은 원소간 곱을 이용하여 / 계산 단위가 filter와 이에 대응되는 윈도우입니다. / 따라서 filter의 size는 / 채널 수 이외의 input의 size에 / 직접적인 간섭을 받지 않습니다.

결과적으로 / Fully Connected 계층은 / 행 또는 열 단위의 증가만 가능해서 / 행렬 형태의 데이터만 처리할 수 있고, / Convolution 계층은 채널 수만큼 연산할 윈도우가 늘어난 셈으로 / 다차원의 데이터를 처리할 수 있습니다.

---

## 7.5 CNN 구현하기

이어서 Convolution 계층이 하나인 / SimpleConvNet을 클래스로 구현하도록 하겠습니다. / 
보이는 것처럼 1층에는 Convolution, ReLU, Pooling이, / 2층에는 Affine과 ReLU로 이루어진 완전 연결 층이, 마지막 3층에는 Affine과 Softmax 계층으로 구성됩니다.

클래스의 선언에서 input의 shape과 convolution에 이용할 매개변수들, / 그리고 2층에 hidden node 갯수, 출력의 label 갯수, / 가중치 초기값에 대한 표준편차를 받습니다. / 
이런 값들을 통해 Convolution 계층의 output shape과 / Pooling 후의 output shape를 구할 수 있습니다. / 이때 pooling은 size를 2로 잡아 / pooling의 출력을 input size의 반으로 만듭니다.

weight과의 연산을 하는 계층을 위해 / 각각의 filter size에 맞춰 임의의 값으로 채워 넣습니다.

<!-- 1층 output: (N, FN, P_OH, P_OW) 또는 (N, FN * P_OH * P_OW) -->
<!-- 2층 output: (N, hidden_size) -->

쉽게 이용하고자 / 앞서 정해뒀던 계층의 형태와 순서대로 / layers라는 순서가 있는 딕셔너리에 저장시킵니다.

나아가 loss를 계산하는 함수와 / 뒤에 gradient descent에 대한 함수까지 / 역전파를 통해 모형을 학습시킨다.

---

## 7.6 CNN 시각화하기

밑에 그림은 학습 전과 후의 weight를 / 이미지로 표현한 겁니다. / 
학습 전 weight은 임의로 뽑힌 초기 가중치였기 때문에 / 그 형태가 어떤 패턴을 알 수 없지만, / train(SimpleNet)을 통해 학습된 가중치를 보면 / 단순한 선들을 나태나고 있습니다. / 
이 단순한 가중치들은 / input 데이터와 곱해져 자신의 형태에 대한 정보를 남깁니다.

### 7.6.2 층 깊이에 따른 추출 정보 변화

다시 / 어떤 이미지에 필터 1과 같이 수직선에 대한 필터를 주면 / 이미지 내에서 수직선에 대한 데이터 값의 신호는 강하게 / 그렇지 않은 데이터는 신호가 작게 표현됩니다. / 
이렇게 weight들과의 연산으로 / filter가 이루는 형태를 input 데이터에서 추출하면 / 다음 층에서 이를 이용할 수 있습니다.

조금 더 확장해보면 / 첫 번쨰 층에서는 단순 엣지, 선분을 검출하고 / 두번째 층에서 선분을 이용한 도형을 인식하고 / 세번쨰 계층에선 전반적인 형태를 파악해서 / 출력의 label을 분류할 수 있게 됩니다.

그래서 Sigmoid를 이용하는 이유처럼 / 층이 깊어질수록 더 복잡한 형태를 인식하게 되고, / 확인하고 비교하는 범위, 단위를 생각해서 추상화된다고 합니다. / 
결과적으로 사물의 의미를 이해하도록 변화하는 것입니다.

---

## 7.7 대표적인 CNN

마지마긍로 이런 CNN에는 대표적으로 LeNet과 AlexNet이 있습니다.

### 7.7.1 LeNet

CNN의 기원이자 첫 CNN인 LeNet은 / 현재의 CNN과는 Activation function이 Sigmoid라는 차이, / max-pooling이 아닌 sub-sampling을 이용했다는 차이가 있습니다.

### 7.7.2 AlexNet

나머지 AlexNet은 / 최근 CNN이라 Activation function으로 ReLU을 이용하며, / 국소적 정규화를 이용하고, / 드랍아웃을 사용한다는 차이가 있습니다.

이상으로 7장 발표를 마칩니다.
