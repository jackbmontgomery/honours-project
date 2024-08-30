import numpy as np

class TransformationFunction():
    def __init__(self):
        pass

    def f(self, x):
        pass

    def df(x):
        pass

class Linear(TransformationFunction):
    def __init__(self):
        pass

    def f(self, x):
       return x

    def df(x):
        return np.full_like(x, 1)