import numpy as np


def identity(x, derivate=False):
    return x if not derivate else np.ones(x.shape)


def logistic(x, derivate=False):
    x[x > 10] = 10
    x[x < -10] = -10

    y = 1 / (1 + np.exp(-x))

    return y if not derivate else y * (1 - y)


def hyperbolic_tangent(x, derivate=False):
    x[x > 10] = 10
    x[x < -10] = -10

    y = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    return y if not derivate else 1 - y * y
