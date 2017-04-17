from PIL import Image
from scipy.fftpack import *
import numpy as np
from utility import *


def gray_scale(image):
    return image.convert("L")


# encode end decode accept ndarray and return ndarray
@profile
def dct1d_codec(arr) -> np.ndarray:
    r, c = np.size(arr, 0), np.size(arr, 1)
    dct1d_cft = dct(arr.flatten(), type=3, norm="ortho")
    return np.reshape(dct1d_cft, (r, c))


@profile
def idct1d_codec(arr) -> np.ndarray:
    r, c = np.size(arr, 0), np.size(arr, 1)
    idct1d_cft = idct(arr.flatten(), type=3, norm="ortho")
    return np.reshape(idct1d_cft, (r, c))


@profile
def dct2d_codec(arr, block_size=(0, 0)) -> np.ndarray:
    assert len(block_size) == 2
    block_rows, block_columns = block_size
    r, c = np.size(arr, 0), np.size(arr, 1)
    if block_rows is 0:
        block_rows = r
    if block_columns is 0:
        block_columns = c
    assert block_rows > 0 and block_columns > 0 and r % block_rows == 0 and c % block_columns == 0
    blocks = blockwise(arr, (block_rows, block_columns))
    dct2d_cft = dct(blocks, type=3, norm="ortho")
    return block_join(dct2d_cft)


@profile
def idct2d_codec(arr, block_size=(0, 0)) -> np.ndarray:
    assert len(block_size) == 2
    block_rows, block_columns = block_size
    r, c = np.size(arr, 0), np.size(arr, 1)
    if block_rows is 0:
        block_rows = r
    if block_columns is 0:
        block_columns = c
    assert block_rows > 0 and block_columns > 0 and r % block_rows == 0 and c % block_columns == 0
    blocks = blockwise(arr, (block_rows, block_columns))
    idct2d_cft = idct(blocks, type=3, norm="ortho")
    return block_join(idct2d_cft)


def psnr(arr_a, arr_b, max_possible=255):
    return 20 * np.math.log10(max_possible) - 10 * np.math.log10(np.mean(np.square(arr_a - arr_b)))


def image_codec_main():
    lena = Image.open("lena.bmp")
    gray_scale_lena = gray_scale(lena)
    gray_scale_lena.save("./output/gray_scale_lena.bmp")
    # test dct1d
    print("=============================================")
    dct1d_cft = dct1d_codec(np.asarray(gray_scale_lena))
    Image.fromarray(dct1d_cft.astype(np.int8), "L").save("./output/dct1d_cft_lena.bmp")
    idct1d_lena = Image.fromarray(idct1d_codec(dct1d_cft).astype(np.int8), "L")
    idct1d_lena.save("./output/idct1d_lena.bmp")
    print("PSNR of 1D DCT codec: %fdB" % psnr(np.asarray(gray_scale_lena), np.asarray(idct1d_lena)))
    # test dct2d

    def test_dct2d(block_size=(0, 0)):
        print("=============================================")
        dct2d_cft = dct2d_codec(np.asarray(gray_scale_lena), block_size)
        Image.fromarray(dct2d_cft.astype(np.int8), "L").save("./output/dct2d_cft_lena_%d%d.bmp" % block_size)
        idct2d_lena = Image.fromarray(idct2d_codec(dct2d_cft, block_size).astype(np.int8), "L")
        idct2d_lena.save("./output/idct2d_lena_%d%d.bmp" % block_size)
        print("PSNR of 2D DCT codec: %fdB, block size: %d, %d" % (psnr(np.asarray(gray_scale_lena), np.asarray(idct2d_lena)), block_size[0], block_size[1]))

    test_dct2d()
    test_dct2d((8, 8))


if __name__ == '__main__':
    image_codec_main()
