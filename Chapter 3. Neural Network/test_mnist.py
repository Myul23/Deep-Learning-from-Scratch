# import sys, os
# sys.path.append(os.pardir)
import numpy as np
from MNIST import *

x, t = neuralnet.get_data()
network = neuralnet.init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = neuralnet.predict(network, x[i])
    p = np.argmax(y) # np.maximum
    if p == t[i]:
        accuracy_cnt += 1

print(f"Accuracy: {str(float(accuracy_cnt) / len(x))}")
# 아니.. 와, 내가 내부 파일 권한까지 준 거 같은데 절대 주소를 줘야 인식하네?
# 여기서 몇 시간을 쓴 거냐 진짜.

# 정규화 (normalization): 데이터를 특정 범위로 변환하는 처리
# 백색화 (whitening): 전체 데이터를 균일하게 분포시킨다 = uniform 분포로 변환? 
# 전처리 (pre-processing): 신경망의 입력 데이터에 특정 변환을 가하는 것

# Batch 처리, Batch: 하나로 묶은 입력 데이터
# 차원 확인 과정
W1, W2, W3 = network["W1"], network["W2"], network["W3"]
print(x.shape, x[0].shape, W1.shape, W2.shape, W3.shape)

# x, t = neuralnet.get_data()
# network = neuralnet.init_network()

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = neuralnet.predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1) # in R, 행기준 which.max
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print(f"Accuracy: {str(float(accuracy_cnt) / len(x))}")
