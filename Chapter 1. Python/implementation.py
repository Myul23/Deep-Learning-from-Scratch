# Jupyter Notebook Update로 VS Code Insiders 사용해야 하는데 아직 확신이 없다.
# 와, 설마 python 전체가 Insiders랑 관련 있는 건가, black formatter가 또 안 먹힌다.

# Chapter 1. 헬로 파이썬
# Numpy, Scipy, Matplotlib
# Caffe, Tensorflow, Chainer, Theano etc.

# 1.2 파이썬 인터프리터
# in cmd python --version | python
# 파이썬 인터프리터: 대화 모드라고 한다?
# 개발자가 물으면(코딩하면) 파이썬 인터프리터가 곧바로 대답한다는 의미

# 1.3 산술연산
# 자료형 타입 알아보기
# class 'int' | 'float' | 'str' | 'bool'
type(10)
type(2.718)
type("hello")

# 변수로의 정의
x = 10
print(x)
x = 100
print(x)
y = 3.14
x * y
type(x * y)

# 리스트(의 등장)
a = [1, 2, 3, 4, 5]
print(a)
len(a)
a[0]
a[4]
a[4] = 99
print(a)
a[0:2]
a[1:]
a[:3]
a[:-1]
a[:-2]

# 딕셔너리
me = {"height": 180}
me["height"]
me["weight"] = 70
print(me)

# bool
hungry = True
sleepy = False
type(hungry)
not hungry
hungry and sleepy
hungry or sleepy

# if
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry\nI'm sleepy")

# for
for i in [1, 2, 3]:
    print(i)


# function
def hello(object="World"):
    print("Hello " + object + "!")


hello()
hello("cat")

# 1.4 파이썬 스크립트 파일
# .py로 저장한 파일에 cmd로 접근해 해당 파일을 열면 함수를 실행시킬 수 있다.


# 클래스
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")
        self.hello()

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")


m = Man("David")
m.goodbye()

# 1.5 Numpy
# 배열 클래스인 numpy.array에는 행렬에 대한 편리한 method가 많다
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x)
type(x)

# 배열은 (행벡터, 열벡터 간의 연산처럼) 원소별(각 자리에 원소 끼리의) 연산을 추구한다.
# x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y, x - y, x * y, x / y, sep=" / ")

# 행렬로의 확장
A = np.array([[1, 2], [3, 4]])
print(A, A.shape, A.dtype, sep=" / ")

B = np.array([[3, 0], [0, 6]])
print(A + B, A * B, sep=" / ")

# broadcast: 스칼라와 행렬 간 연산이 각 원소별 연산으로 확장됨을 의미
# A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
C = np.array([[10, 20], [10, 20]])
print(A * B, A * C, sep=" / ")

# 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X, X[0], X[0][1])

for row in X:
    print(row)

X = X.flatten()
print(X, X[np.array([0, 2, 4])])

print(X > 15, X[X > 15], sep="\n")

# matplotlib
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
# in R, seq(0, 6, by = 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos")

plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

from matplotlib.image import imread

img = imread("C:\dump\HCI\Project1\Project1\Lenna.jpg")
plt.imshow(img)
plt.show()
