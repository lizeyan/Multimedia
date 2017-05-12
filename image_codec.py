from PIL import Image
from scipy.fftpack import *
import numpy as np
from utility import *


def dct1d_codec(arr) -> np.ndarray:
    dct1d_cft = arr.astype(np.float64)
    for axis in range(np.ndim(arr)):
        dct1d_cft = dct(dct1d_cft, type=3, norm="ortho", axis=axis)
    return dct1d_cft


def idct1d_codec(arr) -> np.ndarray:
    idct1d_cft = arr.astype(np.float64)
    for axis in range(np.ndim(arr)):
        idct1d_cft = idct(idct1d_cft, type=3, norm="ortho", axis=axis)
    return idct1d_cft


def dct2d_codec(arr, block_size) -> np.ndarray:
    blocks = blockwise(arr, block_size)
    dct2d_cft = dct(np.swapaxes(dct(np.swapaxes(blocks.astype(np.float64), -1, -2), norm="ortho"), -1, -2), norm="ortho")
    return block_join(dct2d_cft)


def idct2d_codec(arr, block_size) -> np.ndarray:
    blocks = blockwise(arr, block_size)
    idct2d_cft = idct(np.swapaxes(idct(np.swapaxes(blocks.astype(np.float64), -1, -2), norm="ortho"), -1, -2), norm="ortho")
    return block_join(idct2d_cft)


def psnr(arr_a, arr_b, max_possible=255.0):
    return 20 * np.math.log10(max_possible) - 10 * np.log10(np.mean((np.asarray(arr_a, np.float64) - np.asarray(arr_b, np.float64)) ** 2 + np.asarray((1e-10,)), axis=(0, 1)))


def image2arr(image, scale=255) -> np.ndarray:
    return np.asarray(image, dtype=np.float64) / scale


def arr2image(arr: np.ndarray, scale: float = 255) -> Image:
    return Image.fromarray(np.asarray(arr * (scale,)).astype(np.int8), "L")


def image_codec_main():
    output_path = "./output/image_codec"
    clean_folder(output_path)
    lena = Image.open("lena.bmp")
    gray_scale_lena_arr = np.mean(image2arr(lena), axis=-1)
    arr2image(gray_scale_lena_arr).save(os.path.join(output_path, "gray_scale_lena.bmp"))

    for i in range(4):
        test_dct1d(2 ** i, gray_scale_lena_arr, output_path)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    for i in range(4):
        test_dct2d(2 ** i, gray_scale_lena_arr, output_path)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    block_height, block_width = 8, 8
    for i in range(4):
        test_dct2d(2 ** i, gray_scale_lena_arr, output_path, (block_height, block_width))


def test_dct1d(scale, arr, output_path):
    assert np.max(arr) <= 1.0 and np.min(arr) >= 0
    print("=============================================")
    dct1d_cft = dct1d_codec(arr)
    h, w = int(np.size(dct1d_cft, 0) / scale), int(np.size(dct1d_cft, 1) / scale)
    dct1d_cft *= zig_zag_selector(h * w, np.size(arr, 0), np.size(arr, 1))

    Image.fromarray(dct1d_cft.astype(np.int8), "L").save(os.path.join(output_path, "dct1d_cft_lena_%d.bmp" % scale))
    idct1d_lena = arr2image(idct1d_codec(dct1d_cft))
    idct1d_lena.save(os.path.join(output_path, "idct1d_lena_%d.bmp") % scale)
    print("PSNR of 1D DCT codec: %fdB" % psnr(arr * 255, np.asarray(idct1d_lena)), ", use 1/%d DCT coefficients" % scale ** 2)


def test_dct2d(scale, arr, output_path, block_size=(0, 0)):
    print("=============================================")
    if block_size[0] == 0 or block_size[1] == 0:
        block_size = np.shape(arr)

    dct2d_cft = dct2d_codec(arr, block_size)
    dct2d_cft_blocks = blockwise(dct2d_cft, block_size)
    h, w = int(np.size(dct2d_cft_blocks, 2) / scale), int(np.size(dct2d_cft_blocks, 3) / scale)
    dct2d_cft_blocks *= np.expand_dims(np.expand_dims(zig_zag_selector(h * w, block_size[0], block_size[1]), 0), 0)
    dct2d_cft = block_join(dct2d_cft_blocks)

    Image.fromarray((dct2d_cft + 100).astype(np.int8), "L").save(os.path.join(output_path, "dct2d_cft_lena_%d_%d_%d.bmp") % (block_size + (scale, )))
    idct2d_lena = arr2image(idct2d_codec(dct2d_cft, block_size))
    idct2d_lena.save(os.path.join(output_path, "idct2d_lena_%d_%d_%d.bmp") % (block_size[0], block_size[1], scale))
    print("PSNR of 2D DCT codec: %fdB, block size: %d, %d" % (psnr(arr * 255, np.asarray(idct2d_lena)), block_size[0], block_size[1]), ", use 1/%d DCT coefficients" % scale ** 2)


if __name__ == '__main__':
    image_codec_main()
