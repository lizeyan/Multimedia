import os
from PIL import Image
import numpy as np
from numpy.lib.stride_tricks import as_strided
import time
from datetime import datetime
from shutil import *
from numba import *

EPS = 1e-15

__last_tic = None


def tic():
    global __last_tic
    __last_tic = time.time()


def toc():
    global __last_tic
    print("elapsed time: %f s" % (time.time() - __last_tic))
    __last_tic = None


def log(*args, **kwargs):
    print("[%s]" % str(datetime.now())[:-7], *args, **kwargs)


def profile(func):
    def wrap(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        print("Function:", func.__name__, ", Elapsed time:", time.time() - tic, "s")
        return result
    return wrap


def blockwise(matrix, block=(3, 3)):
    return sliding_window(matrix, block, block)


def sliding_window(matrix, block, step=(1, 1)):
    shape = matrix.shape[:-2] + (int((matrix.shape[-2] - block[0]) / step[0] + 1), int((matrix.shape[-1] - block[1]) / step[1] + 1)) + block
    strides = matrix.strides[:-2] + (matrix.strides[-2] * step[0], matrix.strides[-1] * step[1]) + matrix.strides[-2:]
    return as_strided(matrix, shape=shape, strides=strides)


def clean_folder(path):
    if os.path.exists(path):
        assert os.path.isdir(path)
        for file_object in os.listdir(path):
            file_object_path = os.path.join(path, file_object)
            if os.path.isfile(file_object_path):
                os.unlink(file_object_path)
            else:
                rmtree(file_object_path)
    else:
        os.makedirs(path)


def block_join(_blocks):
    blocks = np.rollaxis(_blocks, -1)
    for _ in range(3):
        blocks = np.rollaxis(blocks, -1)
    joined = np.vstack(map(np.hstack, blocks))
    for _ in range(2):
        joined = np.rollaxis(joined, 0, np.ndim(_blocks) - 2)
    return joined


def _test_block():
    # a = np.arange(6 * 6).reshape((6, 6))
    # print(a)
    # print(sliding_window(a, (4, 4), (1, 3)))
    # arr = np.arange(36).reshape((6, 6))
    # blocks = blockwise(arr, (3, 3))
    # print(blocks)
    # re_join = block_join(blocks)
    # print(re_join)
    # print(center_selector(6/8, 8, 8))
    # print(center_selector(4/8, 8, 8))
    # print(center_selector(2/8, 8, 8))
    n = 25, 15
    w = 1024
    h = 512
    a = np.random.randn(*n, w, h)
    print("a.shape", a.shape)
    b = blockwise(a, (8, 8))
    print("b.shape", b.shape)
    a1 = block_join(b)
    print("a1.shape", a1.shape)
    assert np.max(np.abs(a1 - a)) == 0


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


def random_selector(scale, rows, columns):
    result = blockwise(np.zeros((rows, columns)), (scale, scale))
    br, bc = result.shape[0:2]
    idx_each_block = np.unravel_index(np.random.randint(0, scale * scale, size=br * bc), (scale, scale))
    idx_block = np.unravel_index(np.arange(br * bc), (br, bc))
    result[idx_block[0], idx_block[1], idx_each_block[0], idx_each_block[1]] = 1
    return block_join(result)


def center_selector(scale, rows, columns):
    side = (1.0 - scale) / 2
    result = np.ones((rows, columns))
    result[:int(rows * side), :] = 0
    result[-int(rows * side):, :] = 0
    result[:, :int(columns * side)] = 0
    result[:, -int(columns * side):] = 0
    return result


def _test_zig_zag_selector():
    # print(zig_zag_selector(5, 8, 8))
    # print(zig_zag_selector(56, 8, 8))
    print(zig_zag_selector(5, 3, 3))
    print(zig_zag_selector(6, 3, 3))
    print(zig_zag_selector(7, 3, 3))
    print(zig_zag_selector(8, 3, 3))
    print(zig_zag_selector(65, 3, 3))


def image2arr(image: Image, scale: float=255) -> np.ndarray:
    return np.asarray(image, dtype=np.float64) / scale


def arr2image(arr: np.ndarray, scale: float = 255, image_type="L") -> Image:
    return Image.fromarray(np.asarray(arr * (scale,)).astype(np.int8), image_type)


def euclidean_distance(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    return np.sqrt(np.sum((x - y) ** 2, axis=axis))


def histogram_intersection(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    return 1 - np.sum(np.min([x - y + y, y - x + x], axis=0), axis=axis)


def bhattacharyya(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    return np.sqrt(1 + EPS - np.sum(np.sqrt(x * y), axis=axis))


def manhattan_distance(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    return np.sum(np.abs(x - y), axis=axis)


def chebyshev_distance(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    return np.max(np.abs(x - y), axis=axis)


def cosine_distance(_x: np.ndarray, _y: np.ndarray, axis=-1) -> np.ndarray:
    x = _x + np.min(_x)
    y = _y + np.min(_y)
    return 1 - np.sum(x * y, axis=axis) / np.sqrt((np.sum(x * x, axis=axis) * np.sum(y * y, axis=axis)))


def chi_square_distance(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    _x = x + np.min(x)
    _y = y + np.min(y)
    return np.mean((_x - _y) ** 2 / (_x + _y + EPS), axis=axis)


def jffreys_distance(x: np.ndarray, y: np.ndarray, axis=-1) -> np.ndarray:
    return np.sqrt(np.sum((np.sqrt(x + np.min(x)) - np.sqrt(y + np.min(y))) ** 2, axis=axis))


name2func_distance = {
    "l1": manhattan_distance,
    "l2": euclidean_distance,
    "chb": chebyshev_distance,
    "hi": histogram_intersection,
    "bh": bhattacharyya,
    "cos": cosine_distance,
    "ca": chi_square_distance,
    "jf": jffreys_distance,
}


def normalize(arr, axis=-1):
    return arr / np.expand_dims(np.sum(arr, axis=axis), axis=axis)


@vectorize
def add_jit(a, b):
    return a + b


def add(a, b):
    return a + b


def test_jit():
    a = np.random.randn(100, 100, 100)
    b = np.random.randn(100, 100, 100)
    tic()
    for i in range(100):
        add_jit(a, b)
    toc()
    tic()
    for i in range(100):
        add(a, b)
    toc()


if __name__ == '__main__':
    # ignore me
    _test_block()
    # test_jit()
    # _test_zig_zag_selector()
