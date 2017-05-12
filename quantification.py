from image_codec import *
from matplotlib import pyplot

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


def main():
    output_path = "./output/quantification"
    clean_folder(output_path)
    gray_scale_lena = gray_scale(Image.open("lena.bmp"))

    def test_quantifier(q, indicator, alpha=1.0, output=False):
        q = np.asarray(q, dtype=np.float64) * alpha
        block_shape = np.shape(q)
        dct_cft_blockwise = blockwise(dct2d_codec(np.asarray(gray_scale_lena), block_shape), block_shape)
        q_expanded = np.expand_dims(np.expand_dims(q, 0), 0)
        dct_cft_quantified = (dct_cft_blockwise / q_expanded).astype(int) * q_expanded
        idct_quantified = idct2d_codec(block_join(dct_cft_quantified), block_shape)
        psnr_all = psnr(np.asarray(gray_scale_lena), block_join(idct_quantified.astype(np.int8)))
        if output:
            psnr_blockwise = psnr(blockwise(np.asarray(gray_scale_lena), block_shape), blockwise(idct_quantified.astype(np.int8), block_shape))
            np.savetxt(os.path.join(output_path, "psnr.txt"), psnr_blockwise)
            print("PSNR overall: %fdB, %s" % (psnr_all, indicator))
            image = Image.fromarray(block_join(idct_quantified).astype(np.int8), "L")
            image.save(os.path.join(output_path, "dct2d_quantification_%s.bmp" % (indicator,)))
        return psnr_all

    alpha_list = np.arange(0.1, 2, 0.05)
    psnr_list_jpeg = np.vectorize(lambda x: test_quantifier(JPEG_STANDARD_Q, "jpeg", x))(alpha_list)
    psnr_list_nikon = np.vectorize(lambda x: test_quantifier(NIKON_COOLPIX_L12, "nikon", x))(alpha_list)
    psnr_list_canon = np.vectorize(lambda x: test_quantifier(CANON_IXUS_60, "canon", x))(alpha_list)

    pyplot.subplot(311)
    pyplot.plot(alpha_list, psnr_list_jpeg)
    pyplot.ylabel("PSNR (dB")
    pyplot.xlabel("alpha")
    pyplot.title("JPEG")

    pyplot.subplot(312)
    pyplot.plot(alpha_list, psnr_list_nikon)
    pyplot.xlabel("alpha")
    pyplot.ylabel("PSNR (dB")
    pyplot.title("NIKON")

    pyplot.subplot(313)
    pyplot.plot(alpha_list, psnr_list_canon)
    pyplot.xlabel("alpha")
    pyplot.ylabel("PSNR (dB")
    pyplot.title("CANON")

    pyplot.tight_layout()
    pyplot.savefig(os.path.join(output_path, "psnr_alpha.pdf"))


if __name__ == '__main__':
    main()