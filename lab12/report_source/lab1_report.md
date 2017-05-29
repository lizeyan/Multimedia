# Lab1 Report

2014011292 李则言

## DCT/IDCT

### 转化为灰度图

将RGB三个分量取平均数就可以

``` python
lena = Image.open("lena.bmp")
gray_scale_lena_arr = np.mean(image2arr(lena), axis=-1)
arr2image(gray_scale_lena_arr).save(os.path.join(output_path, "gray_scale_lena.bmp"))
```

灰度图：

![gray_lena](../lab1_output/image_codec/gray_scale_lena.bmp)

### PSNR的计算

$$
PSNR=10log_{10}\frac{{MAX}^2}{MSE(a, b)}\\
=20log_{10}{MAX}-10log_{10}MSE(a, b)
$$

``` python
def psnr(arr_a, arr_b, max_possible=255.0):
    return 20 * np.math.log10(max_possible) - 10 * np.log10(np.mean((np.asarray(arr_a, np.float64) - np.asarray(arr_b, np.float64)) ** 2 + (EPS,), axis=(0, 1)))
```



### 1D-DCT（First Row Then Column）

1D-DCT是对一个一维序列做DCT，得到一个一维序列。先行后列的1D-DCT指的是先对图片的每一行做DCT，再对结果的每一列做DCT

1D-DCT的定义为
$$
F(u)=\sqrt{\frac{2}{N}}C_u\sum_{i=0}^{N-1}f(i)cos\frac{(2i+1)u\pi}{2N} \\
\text{where } C_u=\begin{cases}\frac{1}{\sqrt2} & u = 0 \\ 1 & otherwise\end{cases}
$$
那么对于一个M行N列的图片$s_{ij}, i\in[0,M), j \in [0, N)$

对每一行做1D-DCT的结果为$r_{ij}$
$$
r_{ij}=\sqrt{\frac{2}{N}}C_j\sum_{x=0}^{N-1}s_{ix}cos{\frac{(2x+1)j\pi}{2N}}
$$
再对$r_{ij}$每一列做1D-DCT的结果为$t_{ij}$
$$
t_{ij}=\sqrt{\frac{2}{M}}C_i\sum_{y=0}^{M-1}r_{yj}cos{\frac{(2y+1)i\pi}{2M}}\\
=\frac{2}{\sqrt{M\cdot N}}C_iC_j\sum_{x=0}^{M}\sum_{y=0}^{N}s_{xy}cos{\frac{(2x+1)i\pi}{2M}}cos{\frac{(2y+1)j\pi}{2N}}
$$


``` python
def dct1d_codec(arr) -> np.ndarray:
    dct1d_cft = arr.astype(np.float64)
    for axis in range(np.ndim(arr)):
        dct1d_cft = dct(dct1d_cft,  norm="ortho", axis=axis)
    return dct1d_cft.astype(np.int64)
```

将最后的DCT系数保存为整数会减少精度,但是考虑到量化的过程,实际的图片编码过程应该使用整数而不是浮点数来存储这些系数.

得到的DCT系数如下图所示:

![1d_dct](../lab1_output/image_codec/dct1d_cft_lena_1.bmp)

越靠近左上角,即低频部分的DCT系数,值就越大.

1D-IDCT可以将DCT系数恢复为原始图片.对整张图片进行1D-IDCT就是先列后行地执行IDCT.

``` python
def idct1d_codec(arr) -> np.ndarray:
    idct1d_cft = arr.astype(np.float64)
    for axis in reversed(range(np.ndim(arr))):
        idct1d_cft = idct(idct1d_cft, norm="ortho", axis=axis)
    return idct1d_cft
```

恢复出的图片:

![lena_idct](../lab1_output/image_codec/idct1d_lena_1.bmp)

PSNR为49.9988dB,和原图基本没有差别.

### 2D-DCT on the whole image

2D-DCT的定义:
$$
FDCT=\frac{2}{\sqrt{M\cdot N}}C_{u}C_{v}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}s_{ij}cos{\frac{(2i+1)u\pi}{2M}}cos{\frac{(2j+1)v\pi}{2N}} \\
\text{where }C_{i}=\begin{cases}\frac{1}{\sqrt2} & i = 0 \\ 1 & otherwise\end{cases}
$$
根据之前对先行后列的1D-DCT的推导,它和先行后列的1D-DCT是相同的.因此可以这样实现2D-DCT:

``` python
def dct2(arr):
    return dct(np.swapaxes(dct(np.swapaxes(arr, -1, -2), norm="ortho"), -1, -2), norm="ortho")
```

同理,2D-IDCT也可以有类似的实现:

``` python
def idct2(arr):
    return idct(np.swapaxes(idct(np.swapaxes(arr, -1, -2), norm="ortho"), -1, -2), norm="ortho")
```

2D-DCT的DCT系数如下图所示:

![2d_dct_fct](../lab1_output/image_codec/dct2d_cft_lena_512_512_1.bmp)

将其恢复为原始图片的结果为:

![2d_idct_fct](../lab1_output/image_codec/idct2d_lena_512_512_1.bmp)

PSNR为49.9988dB,正如理论分析的结果,和First Row Then Column 1D-DCT/IDCT的结果是相同的.

### 2D-DCT on 8*8 block

和上一节的内容基本相同,只不过现在我们是对图片上每一个8*8的小块分别当成一张单独的小图片进行DCT和IDCT.

首先需要实现对图片进行快速的分块和拼接的方法:

``` python
def blockwise(matrix, block=(3, 3)):  # 分块
    return sliding_window(matrix, block, block)

def sliding_window(matrix, block, step=(1, 1)):
    shape = (int((matrix.shape[0] - block[0]) / step[0] + 1), int((matrix.shape[1] - block[1]) / step[1] + 1)) + block
    strides = (matrix.strides[0] * step[0], matrix.strides[1] * step[1]) + matrix.strides
    return as_strided(matrix, shape=shape, strides=strides)

def block_join(blocks):  # 拼接
    return np.vstack(map(np.hstack, blocks))
```

然后直接在每一个块应用2D-DCT/IDCT:

``` python
def dct2d_codec(arr: np.ndarray, block_size: tuple) -> np.ndarray:
    blocks = blockwise(arr, block_size)
    dct2d_cft = dct2(blocks.astype(np.float64))
    return block_join(dct2d_cft).astype(np.int64)

def idct2d_codec(arr, block_size) -> np.ndarray:
    blocks = blockwise(arr, block_size)
    idct2d_cft = idct2(blocks.astype(np.float64))
    return block_join(idct2d_cft)
```

*这样将表示图片的2维矩阵按8*8分块转换成4维张量进行DCT比使用for循环对每一个块进行DCT是块很多很多倍的,原因一方面是是python的运行机制,另一方面是numpy中对张量运算的优化*

2D-DCT的DCT系数如下图所示:

![2d_dct_fct_88](../lab1_output/image_codec/dct2d_cft_lena_8_8_1.bmp)

将其恢复为原始图片的结果为:

![2d_idct_fct](../lab1_output/image_codec/idct2d_lena_8_8_1.bmp)

PSNR为50.4958dB,和原图基本完全相同.

### 不同方法时间复杂度的比较和分析

### 定义公式和代码实现的复杂度分析

假设输入图片为正方形,边长为$n$

#### 1D

对一个长度为$n$的序列做1D-DCT,需要计算$n$个点,每个点需要$O(n)$的计算量,那么一共需要$O(n^2)$的计算量

那么做先行后列的1D-DCT就一共需要$O(n\times n^2)=O(n^3)$的计算量

实现的时候调用了scipy.fftpack,它不是按照定义实现的,复杂度会小得多,大致为$O(n^2logn)$

#### 2D on the whole image

2D-DCT的定义式中,计算每一个点需要$O(n^2)$的复杂度,所以整张图需要$O(n^4)$的复杂度.

根据之前的理论分析,2D-DCT可以用先行后列的1D-DCT的过程来实现,复杂度可以降低到$O(n^3)$.

考虑到调用scipy.fftpack,复杂度会进一步降低,大致为$O(n^2logn)$

### 2D on 8*8 block

定义式中,每一个块需要$O(8^4)$的复杂度,总共有$(\frac{n}{8})^2$个块,因此复杂度为$O(n^2)$

在实现中,使用先行后列的1D-DCT实现2D-DCT,每一个块需要$O(8^3)$的复杂度,总的复杂度还是$O(n^2)$

考虑到调用scipy.fftpack,每一个块的计算仍然只需常数时间,块的数目不变,总的复杂度还是$O(n^2)$

#### 实现结果

| 时间(s) | 1D-DCT(FRTC) | 2D-DCT(whole) | 2D-DCT(8*8) |
| ----- | ------------ | ------------- | ----------- |
| DCT   | 0.013009     | 0.012007      | 0.011508    |
| IDCT  | 0.009006     | 0.013007      | 0.01501     |
| ALL   | 0.022015     | 0.025015      | 0.026518    |

#### 分析

考虑到这里$n=512$,$O(n^2logn)$和$O(n^2)$并不能反映准确的运行时间的大小关系,实际上常数的影响可能会很大.实验结果也表明了这一点,三种方法的运行时间是没有明确的大小关系的.

### 不同方法PSNR的分析和比较

| METHOD   | dct1d    | dct2d    | dct2d_88 |
| -------- | -------- | -------- | -------- |
| PSNR(dB) | 49.99877 | 49.99877 | 50.49580 |

本质上dct1d和dct2d的实现是相同的,因此它们的PSNR必然是相同的.

dct2d_88比前两者PSNR略高,主要原因我认为在于dct2d_88每个DCT所做的加法次数少,因此计算的舍入误差小,结果就更加精确.

我们可以观察一下它们对应的MSE,分别为0.6504341885642003, 0.5800966545416774. 图片个每个像素取值是$[0, 255)$,这么小的差异用舍入误差完全可以解释.

### 使用一部分DCT系数

#### 原理和实现

对于三种方法,分别使用$\frac{1}{1}, \frac{1}{4}, \frac{1}{16}, \frac{1}{64}$的DCT系数进行IDCT来恢复原始图像.这样做的好处是可以大大压缩图片的占用空间.

这样做的理论依据在之前展示的DCT系数的计算结果中可以看出来,DCT系数的值绝大多数都集中在左上角的低频部分,高频部分基本都在0左右.所以我们可以只取左上角的DCT系数.

具体地将,我们可以用zig-zag顺序选取左上角的DCT系数:

![zig zag](zig_zag.png)

``` python
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
```

上面的函数生成一个binary矩阵,左上角以zig-zag顺序填充1,剩下的是0.只需要将这个矩阵和DCT稀疏矩阵相乘就相当于以zig-zag顺序选取了系数.这样做并没有节省存储空间,但是这样做已经足以说明选取部分DCT系数后恢复出的图片质量,而且实现很简便.

对于2D DCT on 8\*8 block,要在每一个8\*8的块上,而不是全图,来选取DCT系数.

#### 结果与分析

将结果整理如下

| PSNR | dct1d                                    | dct2d                                    | dct2d_88                                 |
| ---- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| 1    | ![lena_idct_1](../lab1_output/image_codec/idct1d_lena_1.bmp)49.99877dB | ![lena_2didct](../lab1_output/image_codec/idct2d_lena_512_512_1.bmp)49.99877dB | ![lena_2didct88_1](../lab1_output/image_codec/idct2d_lena_8_8_1.bmp)50.4958dB |
| 4    | ![lena_idct_2](../lab1_output/image_codec/idct1d_lena_2.bmp)36.09279dB | ![lena_2didct_2](../lab1_output/image_codec/idct2d_lena_512_512_2.bmp)36.09279dB | ![lena_2didct88_2](../lab1_output/image_codec/idct2d_lena_8_8_2.bmp)34.74324dB |
| 16   | ![lena_idct_4](../lab1_output/image_codec/idct1d_lena_4.bmp)29.87871dB | ![lena_2didct_4](../lab1_output/image_codec/idct2d_lena_512_512_4.bmp)29.87871dB | ![lena_2didct88_4](../lab1_output/image_codec/idct2d_lena_8_8_4.bmp)27.85957dB |
| 64   | ![lena_idct_8](../lab1_output/image_codec/idct1d_lena_8.bmp)26.29755dB | ![lena_2didct_8](../lab1_output/image_codec/idct2d_lena_512_512_8.bmp)26.29755dB | ![lena_2didct88_8](../lab1_output/image_codec/idct2d_lena_8_8_8.bmp)24.25341dB |

- 选取的DCT系数越少,图片质量就越低.因为选取的DCT系数越少,高频部分就损失的越严重.

- 如果只选取低频部分,2D-DCT on 8\*8 block的结果比1D-DCT和2D-DCT on the whole image都要差.这是因为如果只保留块内的低频部分,就会使得整张图片颜色的变化不再平滑.在右下角的图片中我们可以很明显地观察到这种现象,每个块内只有一种颜色,块间有明显的颜色跳变.

- PSNR值的大小并不能准确地反映人对图片清晰度的感知.第二行,只选取$\frac{1}{4}$的DCT系数,结果差大约2dB,但是感觉图片质量差不太多.最后一行,同样是差2dB,图片质量差距就非常明显.

  还有更极端的例子,比如下面的图片,和lena.bmp的PSNR有63dB之高,但是内容却只是均一的颜色而已

  ![mean_lena](mean_gray_scale_lena.bmp)

### 代码运行说明

``` bash
python3 image_codec.py
```

- 运行环境:python3.6, 需要numpy, scipy, pillow.
- 大约需要2-3s的运行时间

## Quantization

### 量化的原理

首先我们按照前一个实验的过程,进行2D-DCT on 8\*8 blocks, 得到DCT系数矩阵.

对于DCT系数矩阵中的每一个8\*8的块,使用量化矩阵Q去逐元素地除,注意是整数除法.然后再用Q去和结果矩阵逐元素相乘.量化的一个重要作用就是限制了颜色的数目，因为硬件设备能够处理的颜色是有限的。而且我们人眼能够识别的颜色数目也是有限的，所以量化对图片质量的影响是比较小的。

### 量化矩阵设计

- 因为人眼对图片最直观的感受来自低频部分，高频部分则主要影响图片的细节。而在DCT系数矩阵中，左上角对应低频部分，右上角对应高频部分。所以基本的思路是矩阵中元素的值从左上角往右下角线性递增,这样可以保留低频部分,去除高频部分.线性递增的初始值和增长率经过几次试验确定.

- 量化矩阵中的元素取值越小，PSNR就会越高，但是这种做法就偏离了量化的本意。

- 然后参考给出的几个量化矩阵的设计,右下角高频的部分Q矩阵的值不再增加,也就是锁对于最高频的部分适当保留
  $$
  \text{MY_Q}=\begin{bmatrix}
  1& 1& 1& 2& 2& 3& 4& 4\\
  1& 1& 2& 2& 3& 4& 4& 5\\
  1& 2& 2& 3& 4& 4& 5& 5\\
  2& 2& 3& 4& 4& 5& 5& 6\\
  2& 3& 4& 4& 5& 5& 6& 7\\
  3& 4& 4& 5& 5& 6& 7& 8\\
  4& 4& 5& 5& 6& 7& 8& 7\\
  4& 5& 5& 6& 7& 8& 7& 7\\
  \end{bmatrix}
  $$


### 实验结果与分析

#### 原图和测试图使用不同量化矩阵的PSNR

| JPEG                                     | NIKON                                    | CANON                                    | MY                                       |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| ![jpeg](../lab1_output/quantization/lena_bmp/idct2d_jpeg_1.0.bmp)33.983982dB | ![nikon](../lab1_output/quantization/lena_bmp/idct2d_nikon_1.0.bmp)44.958334dB | ![canon](../lab1_output/quantization/lena_bmp/idct2d_canon_1.0.bmp)45.153871dB | ![my](../lab1_output/quantization/lena_bmp/idct2d_my_1.0.bmp)47.276401dB |

| JPEG                                     | NIKON                                    | CANON                                    | MY                                       |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| ![jpeg](../lab1_output/quantization/shana_jpg/idct2d_jpeg_1.0.bmp)26.146755dB | ![nikon](../lab1_output/quantization/shana_jpg/idct2d_nikon_1.0.bmp)31.262468dB | ![canon](../lab1_output/quantization/shana_jpg/idct2d_canon_1.0.bmp)30.821854dB | ![my](../lab1_output/quantization/shana_jpg/idct2d_my_1.0.bmp)34.185109dB |

#### 原图和测试图的$\alpha-PSNR$曲线

原图:

![lena](../lab1_output/quantization/lena_bmp/psnr_alpha.png)

测试图:

![shana](../lab1_output/quantization/shana_jpg/psnr_alpha.png)

#### 分析

- 对于JPEG使用的Q量化矩阵而言,随着$\alpha$的增大PSNR基本上是在减小,因为保留的信息越来越少。理论上讲，量化后的元素为$\alpha Q_{ij} \times ceil(\frac{a_{ij}}{\alpha Q_{ij}})$，$\alpha$越大，量化后元素可能的取值就越少，有很多颜色的信息就失去了。

  另外三个总体上来说也是在减小，但是在$\alpha=1$附近有不变甚至增加一小段。

- NIKON和CANON都比JPEG的结果要好得多，这两者之间则不相上下。

- 使用测试图片（绘画，非真实场景）和给出的原图（真实照片）进行测试，结论是一致的。

- 自行设计的矩阵在PSNR上和CANON，NIKON，JPEG相比均更好。缺点就是MY_Q中元素的取值比另外三个矩阵都要小，可能的颜色数量要多，对硬件设备的要求会更高。

### 代码运行说明

```bash
python3 quantization.py
```

- 运行环境:python3.6, 需要numpy, scipy, pillow, matplotlib.
- 需要不超过10min的运行时间

## Motion Estimation and Compensation

