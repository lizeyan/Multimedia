import numpy as np
from numpy.lib.stride_tricks import as_strided
import time


def blockwise(matrix, block=(3, 3)):
    shape = (int(matrix.shape[0] / block[0]), int(matrix.shape[1] / block[1])) + block
    strides = (matrix.strides[0] * block[0], matrix.strides[1] * block[1]) + matrix.strides
    return as_strided(matrix, shape=shape, strides=strides)


def block_join(blocks):
    r, c = np.size(blocks, 0), np.size(blocks, 1)
    mat = np.bmat([[np.asmatrix(blocks[row, column]) for column in range(c)] for row in range(r)])
    return np.asarray(mat)


def _test_block():
    arr = np.arange(36).reshape((6, 6))
    blocks = blockwise(arr, (3, 3))
    re_join = block_join(blocks)
    print(re_join)


def profile(func):
    def wrap(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        print("Function:", func.__name__, ", Elapsed time:", time.time() - tic, " ms")
        return result
    return wrap


if __name__ == '__main__':
    _test_block()
