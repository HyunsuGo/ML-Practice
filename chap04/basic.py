import numpy as np

# 평균제곱 오차
def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)

# 교차 엔트로피 오차
def cross_entropy_error0(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

# 미니배치 학습
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import  load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, 10)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 배치용 교차 엔트로피 오차
def cross_entropy_error1(y, t): #원 핫 인코딩시
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size

def cross_entropy_error2(y, t): #숫자레이블 일때
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 수치미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x-h)+f(x+h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# 편미분
def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x*0 + 4.0**2.0
numerical_diff(function_tmp1, 3.0)

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1
numerical_diff(function_tmp2, 4.0)


# 기울기
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] =tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))


# 경사 하강법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))

init_x = np.array([-3.0, 4.0]) #학습률이 너무 큰 예
print(gradient_descent(function_2, init_x, lr=10.0, step_num=100))

init_x = np.array([-3.0, 4.0]) # 학습률이 너무 작은 예
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
