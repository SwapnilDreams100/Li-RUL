import numpy as np
import math

## Kernel Implementation
def polynomial(gamma, c, d):
    def kernel(u, v):
        return (gamma * np.dot(u, v) + c)**d
    return kernel

def rbf(gamma):
    def kernel(u, v):
        w = u - v
        return math.exp( -gamma * np.dot(w, w))
    return kernel