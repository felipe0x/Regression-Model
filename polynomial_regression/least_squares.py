import numpy as np


def Vandermonde(x, d):
    N = x.size
    A = np.zeros((N, d+1))

    for i in range(d+1):
        A[:, i] = x ** i

    return A

def solver(A, y):
    inv_gram = np.linalg.inv(np.transpose(A) @ A)
    theta = inv_gram @ np.transpose(A) @ y

    return theta