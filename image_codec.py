from PIL import *
from PIL import Image
from scipy.fftpack import *
import numpy as np


def gray_scale(image):
    return image.convert("L")


# encode end decode accept ndarray and return ndarray
def dct1d_codec(arr) -> np.ndarray:
    pass


def idct1d_codec(arr) -> np.ndarray:
    pass


def dct2d_codec(arr, block_size) -> np.ndarray:
    pass


def idct2d_codec(arr, block_size) -> np.ndarray:
    pass


def main():
    lena = Image.open("lena.bmp")
    gray_scale_lena = gray_scale(lena)
    gray_scale_lena.save("./output/gray_scale_lena.bmp")




if __name__ == '__main__':
    main()
