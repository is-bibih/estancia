from copy import Error
import numpy as np
import mpmath as mp

def integer_stream(x0, xf, block_size=100, shape=[-1]):
    """Produce an iterator of integer arrays.

    :x0: initial value
    :xf: final value for x
    :block_size: size for outputs
    :shape: shape for iteration variable array
    :source: https://stackoverflow.com/questions/31439875/infinite-summation-in-python

    """
    x = np.arange(x0, x0+block_size).reshape(shape)
    while x.flatten()[-1] <= xf:
        yield x
        x += block_size

def converge_sum(f, x_stream, eps=1e-5, axis=0, max_iter=1e5, **kwargs):
    """Do a summation until it converges or max iterations are reached.

    :f: function to sum
    :x_stream: integer stream to evaluate sums
    :eps: convergence tolerance
    :axis: axis to perform summation on
    :max_iter: cap for iterations
    :source: https://stackoverflow.com/questions/31439875/infinite-summation-in-python

    """
    # holds sum
    total = np.sum(f(next(x_stream)), axis=axis, **kwargs)
    # iteration counter
    i = 0
    # add terms for each block of x values
    x_block = next(x_stream, None)
    while (x_block is not None):
        i += 1
        # contribution from current block
        try:
            diff = np.sum(f(x_block), axis=axis, **kwargs)
            # stop if norm of contribution is small or invalid
            contrib = float(mp.mpf(np.real(np.abs(diff.sum()))))
            if np.isnan(contrib) or (contrib <= eps) or (i > max_iter):
                return total
            else:
                total += diff
            x_block = next(x_stream, None)
        except TypeError as e:
            print(type(contrib))
        except Error as e:
            return total

