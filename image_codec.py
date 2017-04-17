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
    output_path = "./output/image_codec"
    clean_folder(output_path)
    lena = Image.open("lena.bmp")
    gray_scale_lena = gray_scale(lena)
    gray_scale_lena.save(os.path.join(output_path, "gray_scale_lena.bmp"))
    w, h = gray_scale_lena.size
    gray_scale_lena_size = w * h
    selectors = [zig_zag_selector(gray_scale_lena_size / (4 ** i), h, w) for i in range(4)]
    for idx, selector in enumerate(selectors):
        test_dct1d(selector, "1/%d" % (4 ** idx), gray_scale_lena, output_path)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    for idx, selector in enumerate(selectors):
        test_dct2d(selector, "1/%d" % (4 ** idx), gray_scale_lena, output_path)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    selectors = [zig_zag_selector(64 / (4 ** i), 8, 8) for i in range(4)]
    for idx, selector in enumerate(selectors):
        test_dct2d(selector, "1/%d" % (4 ** idx), gray_scale_lena, output_path, (8, 8))


def test_dct1d(selector, selector_indicator, image, output_path):
    print("=============================================")
    dct1d_cft = dct1d_codec(np.asarray(image))
    dct1d_cft *= selector
    Image.fromarray(dct1d_cft.astype(np.int8), "L").save(os.path.join(output_path, "dct1d_cft_lena.bmp"))
    idct1d_lena = Image.fromarray(idct1d_codec(dct1d_cft).astype(np.int8), "L")
    idct1d_lena.save(os.path.join(output_path, "idct1d_lena_%s.bmp") % selector_indicator.replace("/", "_"))
    print("PSNR of 1D DCT codec: %fdB" % psnr(np.asarray(image), np.asarray(idct1d_lena)), ", use %s DCT coefficients" % selector_indicator)


def test_dct2d(selector, selector_indicator, image, output_path, block_size=(0, 0)):
    print("=============================================")
    if block_size[0] == 0 or block_size[1] == 0:
        block_size = image.size
    dct2d_cft = dct2d_codec(np.asarray(image), block_size)
    dct2d_cft_blocks = blockwise(dct2d_cft, block_size)
    dct2d_cft_blocks *= np.expand_dims(np.expand_dims(selector, 0), 0)
    dct2d_cft = block_join(dct2d_cft_blocks)
    Image.fromarray(dct2d_cft.astype(np.int8), "L").save(os.path.join(output_path, "dct2d_cft_lena_%d_%d.bmp") % block_size)
    idct2d_lena = Image.fromarray(idct2d_codec(dct2d_cft, block_size).astype(np.int8), "L")
    idct2d_lena.save(os.path.join(output_path, "idct2d_lena_%d_%d_%s.bmp") % (block_size[0], block_size[1], selector_indicator.replace("/", "_")))
    print("PSNR of 2D DCT codec: %fdB, block size: %d, %d" % (psnr(np.asarray(image), np.asarray(idct2d_lena)), block_size[0], block_size[1]), ", use %s DCT coefficients" % selector_indicator)


if __name__ == '__main__':
    image_codec_main()
