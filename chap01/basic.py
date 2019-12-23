# 자료형
print(type(10))
print(type(2.718))
print(type("hello"))

# 변수
x=10
print(x)

y=3.14
print(x*y)
print(type(x*y))

# 리스트
a = [1, 2, 3, 4, 5]
print(a)
print(len(a))
print(a[0])
a[4]=99
print(a)
print(a[0:2])
print(a[1:])
print(a[:3])
print(a[:-1])
print(a[:-2])

# 딕셔너리
me = {'height' : 180}
me['height']
me['weight']=70 # 새 원소 추가
print(me)

# bool
hungry=True
sleepy=False
print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

# if문
if hungry:
    print("I'm hungry")

hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

# for문
for i in [1, 2, 3]:
    print(i)

# 함수
def hello():
    print("Hello World!")
hello()

def hello(object):
    print("Hello " + object + "!")
hello("cat")

# 클래스
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

# 넘파이
import  numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# 넘파이의 산술연산 (원소수가 같아야함:element-wise)
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)

# 넘파이의 N차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape) # 행렬의 형상
print(A.dtype) # 원소의 자료형
print(A*10)

# 브로드 캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A*B)

# 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)

X = X.flatten() # X를 1차원 배열로 변환(평탄화)
print(X)
print(X[np.array([0, 2, 4])])

print(X>15)
print(X[X>15])

# matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()


# pyplot의 기능
import  numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()


# 이미지 표시하기
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('lena.png')
plt.imshow(img)
plt.show()
