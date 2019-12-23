import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a

