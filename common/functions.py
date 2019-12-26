import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

def cross_entropy_error(y, t):
    if y.ndim==1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
