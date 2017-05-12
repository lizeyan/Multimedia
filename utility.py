import os

import numpy as np
from numpy.lib.stride_tricks import as_strided
import time
from shutil import *


def profile(func):
    def wrap(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        print("Function:", func.__name__, ", Elapsed time:", time.time() - tic, " ms")
        return result
    return wrap


def blockwise(matrix, block=(3, 3)):
    shape = (int(matrix.shape[0] / block[0]), int(matrix.shape[1] / block[1])) + block
    strides = (matrix.strides[0] * block[0], matrix.strides[1] * block[1]) + matrix.strides
    return as_strided(matrix, shape=shape, strides=strides)


def clean_folder(path):
    if os.path.exists(path):
        assert os.path.isdir(path)
        rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def block_join(blocks):
    return np.vstack(map(np.hstack, blocks))


def _test_block():
    arr = np.arange(36).reshape((6, 6))
    blocks = blockwise(arr, (3, 3))
    print(blocks)
    re_join = block_join(blocks)
    print(re_join)


def left_top_corner_selector(r, c, ar, ac):
    ret = np.zeros(shape=(ar, ac), dtype=int)
    ret[:int(r), :int(c)] = 1
    return ret


def zig_zag_selector(length, rows, columns):
    assert rows > 0 and columns > 0 and length > 0
    if length >= rows * columns:
        return np.ones(shape=(rows, columns), dtype=int)
    ret = np.zeros(shape=(rows, columns), dtype=int)
    last = np.asarray((0, 0))
    cnt = 0
    adder_tuple = [(-1, 1), (1, -1)]
    adder = adder_tuple[1]
    line_cnt = 0
    length = min(length, rows * columns)
    line_start = False
    while True:
        ret[last[0], last[1]] = 1
        cnt += 1
        if cnt >= length:
            break
        if not line_start and (last[0] == 0 or last[1] == 0 or last[1] == columns - 1 or last[0] == rows - 1):
            if line_cnt % 2 == 0:
                last = last + (1, 0) if last[1] == columns - 1 else last + (0, 1)
            else:
                last = last + (0, 1) if last[0] == rows - 1 else last + (1, 0)

            line_start = True
            line_cnt += 1
            adder = adder_tuple[line_cnt % 2]
        else:
            line_start = False
            last += adder

    return ret


def _test_zig_zag_selector():
    # print(zig_zag_selector(5, 8, 8))
    # print(zig_zag_selector(56, 8, 8))
    print(zig_zag_selector(5, 3, 3))
    print(zig_zag_selector(6, 3, 3))
    print(zig_zag_selector(7, 3, 3))
    print(zig_zag_selector(8, 3, 3))
    print(zig_zag_selector(65, 3, 3))


if __name__ == '__main__':
    _test_block()
    # _test_zig_zag_selector()
