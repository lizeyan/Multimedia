from image_codec import *
from matplotlib import pyplot
import cProfile

CANON_IXUS_60 = np.asarray([
    [1,  1,  1,  2,  3,  6,  8,  10],
    [1,  1,  2,  3,  4,  8,  9,  8],
    [2,  2,  2,  3,  6,  8,  10, 8],
    [2,  2,  3,  4,  7,  12, 11, 9],
    [3,  3,  8,  11, 10, 16, 15, 11],
    [3,  5,  8,  10, 12, 15, 16, 13],
    [7,  10, 11, 12, 15, 17, 17, 14],
    [14, 13, 13, 15, 15, 14, 14, 14],
])

NIKON_COOLPIX_L12 = np.asarray([
    [2, 1,  1,  2,  3,  5,  6,  7],
    [1, 1,  2,  2,  3,  7,  7,  7],
    [2, 2,  2,  3,  5,  7,  8,  7],
    [2, 2,  3,  3,  6,  10, 10, 7],
    [2, 3,  4,  7,  8,  13, 12, 9],
    [3, 4,  7,  8,  10, 12, 14, 11],
    [6, 8,  9,  10, 12, 15, 14, 12],
    [9, 11, 11, 12, 13, 12, 12, 12],
])

JPEG_STANDARD_Q = np.asarray([
    [16, 11, 10, 16, 24,  40,  51, 61],
    [12, 12, 14, 19, 26,  58,  60, 55],
    [14, 13, 16, 24, 40,  57,  69, 56],
    [14, 17, 22, 29, 51,  87,  80, 62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 12,  100, 103, 99],
])


@profile
def main():
    output_path = "./output/quantification"
    clean_folder(output_path)
    jpeg = np.expand_dims(np.expand_dims(JPEG_STANDARD_Q, 0), 0).astype(np.float64)
    nikon = np.expand_dims(np.expand_dims(NIKON_COOLPIX_L12, 0), 0).astype(np.float64)
    canon = np.expand_dims(np.expand_dims(CANON_IXUS_60, 0), 0).astype(np.float64)
    gray_scale_lena_arr = np.mean(image2arr(Image.open("lena.bmp")), axis=-1)

    def test_quantifier(q: np.ndarray, indicator: str, output=False):
        dct_cft_quantified = dct2d_codec(gray_scale_lena_arr, block_shape, q)
        idct_quantified = idct2d_codec(dct_cft_quantified, block_shape)

        psnr_all = psnr(gray_scale_lena_arr, block_join(idct_quantified), 1.0)
        if output:
            psnr_blockwise = psnr(blockwise(gray_scale_lena_arr, block_shape), blockwise(idct_quantified, block_shape), 1.0)
            np.savetxt(os.path.join(output_path, "%s_psnr.txt" % (indicator,)), psnr_blockwise)
            print("PSNR overall: %fdB, %s" % (psnr_all, indicator))
            image = arr2image(block_join(idct_quantified))
            image.save(os.path.join(output_path, "dct2d_quantification_%s.bmp" % (indicator,)))
        return psnr_all

    test_quantifier(jpeg, "jpeg_1.0", True)
    test_quantifier(nikon, "nikon_1.0", True)
    test_quantifier(canon, "canon_1.0", True)

    alpha_list = np.arange(0.1, 2, 0.05)
    psnr_list_jpeg = np.vectorize(lambda x: test_quantifier(jpeg * x, "jpeg_%f" % x))(alpha_list)
    psnr_list_nikon = np.vectorize(lambda x: test_quantifier(nikon * x, "nikon_%f" % x))(alpha_list)
    psnr_list_canon = np.vectorize(lambda x: test_quantifier(canon * x, "canon_%f" % x))(alpha_list)

    pyplot.subplot(311)
    pyplot.plot(alpha_list, psnr_list_jpeg)
    pyplot.ylabel("PSNR (dB")
    pyplot.xlabel("alpha")
    pyplot.title("JPEG")
    print("JPEG finished")

    pyplot.subplot(312)
    pyplot.plot(alpha_list, psnr_list_nikon)
    pyplot.xlabel("alpha")
    pyplot.ylabel("PSNR (dB")
    pyplot.title("NIKON")
    print("NIKON finished")

    pyplot.subplot(313)
    pyplot.plot(alpha_list, psnr_list_canon)
    pyplot.xlabel("alpha")
    pyplot.ylabel("PSNR (dB")
    pyplot.title("CANON")
    print("CANON finished")

    pyplot.tight_layout()
    pyplot.savefig(os.path.join(output_path, "psnr_alpha.pdf"))


if __name__ == '__main__':
    block_shape = (8, 8)
    main()
