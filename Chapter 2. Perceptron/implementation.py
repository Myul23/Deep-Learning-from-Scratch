# Chapter 2. 퍼셉트론
import numpy as np

# 2.3 퍼셉트론의 구현
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = b + np.sum(w * x)
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0

print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = b + np.sum(w * x)
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0

print(NAND(0, 0), NAND(0, 1), NAND(1, 0), NAND(1, 1))

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = b + np.sum(w * x)
    if tmp > 0:
        return 1
    elif tmp <= 0:
        return 0

print(OR(0, 0), OR(0, 1), OR(1, 0), OR(1, 1))

# 2.4 퍼셉트론의 한계: 단일 선형 조합의 한계
# 2.5 다층 퍼셉트론: 층을 쌓다, 퍼셉트론 조합과 비선형 분리의 시작

def XOR(x1, x2):
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    return AND(s1, s2)

print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))
