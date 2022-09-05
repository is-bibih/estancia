import mpmath as mp
import numpy as np

def M(a, b, z):
    """Wrap mpmath's 1F1 for numpy arrays.

    :a: first parameter
    :b: second parameter
    :z: ndarray argument
    :returns: M(a, b, z) as a ndarray

    """
    sh = z.shape
    y = [mp.hyp1f1(a, b, zi) for zi in z.flatten()]
    y = np.array(y).reshape(sh)
    return y

def U(a, b, z):
    """Wrap mpmath's hyperu for numpy arrays.

    :a: first parameter
    :b: second parameter
    :z: ndarray argument
    :returns: U(a, b, z) as a ndarray

    """
    sh = z.shape
    y = [mp.hyperu(a, b, zi) for zi in z.flatten()]
    y = np.array(y).reshape(sh)
    return y

