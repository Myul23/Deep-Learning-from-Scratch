# [Chapter 5. 오차역전법](https://www.miricanvas.com/design/17uine)

## 5.5 활성화 함수 계층 구현하기

### 5.5.1 ReLU 계층

제일 먼저 ReLU입니다. / 
ReLU는 / 입력으로 들어온 값이 0 이하일 때만 / 0으로 출력하고 / 나머지는 입력값을 그대로 보냅니다.

이는 입력값이 0보다 작을 때는 / 출력에 아무런 영향을 주지 못한다는 것이죠. / 
그리고 0보다 큰 값에 대해선 / 다른 연산을 하지 않으므로 그대로 흘려보냅니다.

코드도 이와 크게 다르지 않습니다. / 
이 코드는 / ReLU를 class로 구현한 코드입니다. / 
순전파 때 / 0을 기준으로 하는 / 입력값에 대한 mask를 만들어둡니다. / 
이를 역전파 때 이용하는 방식입니다.

### 5.5.2 Sigmoid 계층

다음으로 Sigmoid 입니다. / 
Sigmoid는 / 입력값에 -1를 곱하고 / exponential을 취한 값에 / 1을 더하고 / 마지막으로 역수를 취해 / 구합니다.

각각을 하나의 독립된 식으로 봤을 때 / 미분 값을 이용한 연쇄작용으로 / 대응하는 값을 곱해주는 형식입니다.

마지막으로 / 전체 출력 y에 대한 함수로 형태를 바꾸면 / sigmoid 함수의 역전파값은 / y * (1 - y)로 축약됩니다.

코드도 여기서 크게 달라지지 않습니다.

---

## 5.6 Affine/Softmax 계층 구현하기

### 5.6.1 Affine 계층

다음은 Affine 변환입니다. / Affine 변환은 행렬곱을 뜻합니다.

행렬곱의 미분 결과는 / 입력값의 모든 자리 마다의 미분과 같으므로 / 입력값의 형상과 같아야 하고 / 곱셈을 포함하기 때문에 / 다른 피연산자의 값을 곱해야 합니다.

### 5.6.2 배치용 Affine 계층

코드에 역전파 함수를 보시면 / 단순 덧셈 노드를 통한 편향에도 / 역전파 값을 전달하고 있습니다. / 
가중치와 마찬가지로 / 편향도 기존의 shape을 유지하면서 / 연산된 각 위치의 역전파 값을 더해 만듭니다.

### 5.6.3 Softmax-with-Loss 계층 (p294 - p298)

다음은 소프트맥스를 / 손실 함수와 함께 구현하는 것입니다.

먼저 손실 함수로 쓰인 Cross-Entropy Error를 역방향으로 보면, / 1로 출발해 -1을 곱하고

덧셈 노드이므로 같은 값으로 흐르며

각 tk와 곱해진 후

log 미분이 입력에 역수이므로 / yk의 역수를 넘긴다.

다음 Softmax입니다. / 
여기서 S는 exp(ak)의 총합을 의미하고 / 처음 곱하기 노드에 따라 / exp(ak)를 곱한 값을 보냅니다. / 
여기서 yk가 exp(ak)를 exp(ak)의 총합으로 나눈 것이라 / -tk와 S의 곱이 전해집니다. / 

거기에 원-핫 인코딩에 따라 tk의 총합이 1이되고 / -1승의 미분은 입력값을 1/s일 때 - 1/s^2이므로 / 서로 상쇄가 되어 / 위쪽 흐름은 1/s로

그대로 전달됩니다.

exponential 이전에 아래쪽 흐름은 -tk / exp(ak)가 되고 / 
이 값과 1/S 그리고 exp 노드의 입력값 exp(ak)와 곱하면 / 결과적으로 Cross-Entropy-Error 계층과 Softamx 계층은 예측과 실제 사이의 차이를 / 역전파 값으로 전달한다는 것을 알 수 있습니다. / 
더불어 항등함수의 손실함수로 오차제곱합을 사용하는 것도 Softmax & Cross-Entropy-Error 혼합 계층과 같이 / 예측과 실제값의 차이를 역전파 값으로 전달하기 때문입니다.

이를 코드로 구현하면 / 이전에 정의한 함수의 이용과 / 예측값과 실제값의 차이로 축약할 수 있습니다.

---

## 5.7 오차역전파법 구현하기

### 5.7.1 신경망 학습의 전체 그림

지금까지의 신경망 학습을 되돌아보면 / 주어진 데이터에 대해 미니배치를 구성하고 / 해당 데이터로 기울기를 산출해 / 학습률의 곱으로 매개변수를 업데이트합니다. / 그리고 앞선 방법들을 특정 기준을 만족할 때까지 반복합니다.

### 5.7.2 오차역전파법을 적용한 신경망 구현하기

그렇다면 지금까지한 역전파를 / 앞선 장에서 구현한 2층 신경망에 추가해 / 구현해보도록 하겠습니다. / 
init 함수에서 / 이 신경망에서 사용할 모든 층과 매개변수를 선언합니다. / 
앞선 ReLU, Sigmoid, Affine을 이용할 수 있는데 / 여기서는 OrderedDict를 통해 / 입력된 순서대로 순위를 부여합니다. 
그리고 마지막 층에는 SoftmaxWithLoss를 통해 / 출력을 범주로 한정하고 / Loss를 구합니다.

뒤이어 predict, loss, accuracy 함수를 통해 / 모형의 예측값과 loss, 정확도를 계산할 수 있습니다.

이어 gradient descent 계산에는 / params 변수의 값을 불러와 이용하며 / 더 빠른 계산을 위해 역전파를 이용해 계산합니다.

### 5.7.3 오차역전파법으로 구한 기울기 검증하기

이렇게 역전파를 이용해 구한 기울기가 / 수치 미분으로 계산한 값과 얼마만큼의 차이가 있을까요? / 
앞선 2ㄴ층 신경망 클래스에서 / 수치 미분에 대한 함수 또한 구현하였으므로 / 이를 이용하면 됩니다. / 
ppt에 결과를 싣지는 않았지만, / 그 차이가 numpy 패키지의 표기로 확인했을 때 0으로 표현함을 확인하였습니다.

### 5.7.4 오차역전파법을 사용한 학습 구현하기

그래서 이를 이용해 앞선 장에서처럼 / train, test의 loss 및 정확도를 담아줄 변수를 선언하거나 / 배치를 구성하는 등으로 이용할 수 있습니다.

/ 이상으로 발표를 마칩니다.
