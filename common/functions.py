import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

def cross_entropy_error(y, t):
    if y.ndim==1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size
