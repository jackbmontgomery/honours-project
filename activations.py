import numpy as np

def linear(x):
    return x

def linear_derivative(x):
    return np.full_like(x, 1)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def sigmoid(x):
    return 1 / ( 1 + np.e ** (- x))

def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))