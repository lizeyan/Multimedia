# 声音转换实验

2014011292 李则言

---

## 实验一

### 实验内容

1. 将某非著名相声演员的声音转换为3种不同的声音
2. 将著名评书演员单田芳的声音转换为3种不同的声音

## 实验结果

使用提供的straight软件，基于MATLABr2017b运行。

直接修改若干参数，得到不同的音频文件即可。

修改步骤：

1. 打开matlab
2. 将straight所在的路径添加到`路径`中（只需要操作一次）
3. 在命令行中输入straight，打开工具的GUI
4. initialize
5. read from file
6. analyse source
7. analyse MBX
8. bypass
9. 修改左下角的参数，只能拖动滑动条或者点击箭头，直接修改数值没有用
10. synthsize grad
11. play synthsized
12. save to file

### `guodegang.wav`

| 文件名            | F0     | Frequency | Temporal | 描述                          |
| -------------- | ------ | --------- | -------- | --------------------------- |
| guodegang1.wav | 1.5336 | 1         | 1        | 声音变高变尖变明亮，但是还是男性的声音         |
| guodegang2.wav | 1      | 1.2932    | 0.72303  | 变成了成年女性的声音，声音不高，稍微有些嘶哑，语速变快 |
| guodegang3.wav | 2.5119 | 1.4423    | 1.067    | 声音变得特别明亮，像一个孩子的声音           |

### `shantianfang.wav`

| 文件名               | F0      | Frequency | Temporal | 描述                |
| ----------------- | ------- | --------- | -------- | ----------------- |
| shantianfang1.wav | 0.90603 | 0.91786   | 1        | 变成了一个壮年男性的声音，更加低沉 |
| shantianfang2.wav | 2.8651  | 1.2732    | 0.87834  | 变成了声音很高的女声        |
| shantianfang3.wav | 0.93633 | 1.2151    | 0.87834  | 变成了一个声音低的老年女声     |

### 结论

F0参数表示基频。基频越高声音听起来就越高，越明亮。

Frequency表示频率，女声和童声的频率都比较高。

Temporal表示语速。

## 实验二

### 实验任务

将A的声音转换为B的声音

### 实验工具

使用提供的straight软件，基于MATLABr2017b运行。使用提供的praat软件分析基频和共振峰以及时长，得到调整时使用的参数。

### 基频分析的步骤

1. 打开praat

2. 在`read`->`read from file`打开音频文件

3. 选择`Analyse`->`Periodicity`，进行基频分析

   ![pitch1](./data/pitch1.png)

4. 选择`convert`->`down to pitchtier`

   ![pitch2](./data/pitch2.png)

5. 选择`Edit`，查看分析结果

   ![pitch3](./data/pitch3.png)

   ![pitch4](./data/pitch4.png)

   ​

### 共振峰分析的步骤

1. 打开praat

2. 在`read`->`read from file`打开音频文件

3. 选择`edit`，查看分析结果

   ![formant1](./data/formant1.png)

4. 选择`formant`->`formant listing`，查看共振峰

   ![formant2](./data/formant2.png)

### 实验结果

#### `sen6000.wav`

|      | 基频                                       | 共振峰                                      | 基频最大值  | 基频最小值  | 共振峰1        | 共振峰2        | 共振峰3        | 共振峰4        | Time |
| ---- | ---------------------------------------- | ---------------------------------------- | ------ | ------ | ----------- | ----------- | ----------- | ----------- | ---- |
| A    | ![sen6000pitch](./data/实验2/A/sen6000pitch.png) | ![sen6000pitch](./data/实验2/A/sen6000formant.png) | 177.39 | 76.20  | 644.763986  | 1363.739241 | 2904.370424 | 3644.749440 | 7.73 |
| B    | ![sen6000pitch](./data/实验2/B/sen6000pitch.png) | ![sen6000pitch](./data/实验2/B/sen6000formant.png) | 389.5  | 141.10 | 1953.651604 | 2572.033129 | 3509.947900 | 4283.263562 | 7.76 |

- 取基频最大值和最小值平均值的比值作为F0参数,最大值和最小值排除了图中特别大和特别小的点

- 取频率最高的F4共振峰的比值作为frequency参数

- 取时长的比值作为temporal参数

- 由于straight软件的限制，不能自由输入参数的值。下列的参数和理论上的结果有微小的不同。

  | 文件名           | F0     | Frequncy | Temporal |
  | ------------- | ------ | -------- | -------- |
  | sen6000AB.wav | 2.0756 | 1.1778   | 1.033    |

#### `sen6015.wav`

|      | 基频                                       | 共振峰                                      | 基频最大值  | 基频最小值  | 共振峰1       | 共振峰2        | 共振峰3        | 共振峰4        | Time |
| ---- | ---------------------------------------- | ---------------------------------------- | ------ | ------ | ---------- | ----------- | ----------- | ----------- | ---- |
| A    | ![sen6015pitch](./data/实验2/A/sen6015pitch.png) | ![sen6015pitch](./data/实验2/A/sen6015formant.png) | 191.30 | 75.07  | 783.357741 | 1978.990360 | 2779.413316 | 4064.984247 | 5.40 |
| B    | ![sen6015pitch](./data/实验2/B/sen6015pitch.png) | ![sen6015pitch](./data/实验2/B/sen6015formant.png) | 440.8  | 149.80 | 706.058900 | 1926.658355 | 3203.342404 | 4436.170193 | 5.30 |

- 取基频最大值和最小值平均值的比值作为F0参数,最大值和最小值排除了图中特别大和特别小的点

- 取频率最高的F4共振峰的比值作为frequency参数

- 取时长的比值作为temporal参数

- 由于straight软件的限制，不能自由输入参数的值。下列的参数和理论上的结果有微小的不同。

  | 文件名           | F0     | Frequncy | Temporal |
  | ------------- | ------ | -------- | -------- |
  | sen6015AB.wav | 2.2022 | 1.0895   | 0.98136  |

#### `sen6028.wav`

|      | 基频                                       | 共振峰                                      | 基频最大值  | 基频最小值 | 共振峰1        | 共振峰2        | 共振峰3        | 共振峰4        | Time |
| ---- | ---------------------------------------- | ---------------------------------------- | ------ | ----- | ----------- | ----------- | ----------- | ----------- | ---- |
| A    | ![sen6028pitch](./data/实验2/A/sen6028pitch.png) | ![sen6028pitch](./data/实验2/A/sen6028formant.png) | 205.8  | 74.97 | 1346.221414 | 1776.228071 | 3228.814884 | 4133.926421 | 9.02 |
| B    | ![sen6028pitch](./data/实验2/B/sen6028pitch.png) | ![sen6028pitch](./data/实验2/B/sen6028formant.png) | 421.16 | 76.21 | 1082.817883 | 2283.150850 | 3496.125188 | 4947.162751 | 8.09 |

- 取基频最大值和最小值平均值的比值作为F0参数,最大值和最小值排除了图中特别大和特别小的点

- 取频率最高的F4共振峰的比值作为frequency参数

- 取时长的比值作为temporal参数

- 由于straight软件的限制，不能自由输入参数的值。下列的参数和理论上的结果有微小的不同。

  | 文件名           | F0     | Frequncy | Temporal |
  | ------------- | ------ | -------- | -------- |
  | sen6028AB.wav | 1.7841 | 1.2151   | 0.89039  |

#### `sen6044.wav`

|      | 基频                                       | 共振峰                                      | 基频最大值 | 基频最小值  | 共振峰1        | 共振峰2        | 共振峰3        | 共振峰4        | Time |
| ---- | ---------------------------------------- | ---------------------------------------- | ----- | ------ | ----------- | ----------- | ----------- | ----------- | ---- |
| A    | ![sen6044pitch](./data/实验2/A/sen6044pitch.png) | ![sen6044pitch](./data/实验2/A/sen6044formant.png) | 191.8 | 75.02  | 1006.356758 | 1798.415307 | 3167.092465 | 3974.637219 | 6.19 |
| B    | ![sen6044pitch](./data/实验2/B/sen6044pitch.png) | ![sen6044pitch](./data/实验2/B/sen6044formant.png) | 349.7 | 146.79 | 881.822380  | 2326.346961 | 3409.009971 | 4264.112235 | 5.47 |

- 取基频最大值和最小值平均值的比值作为F0参数,最大值和最小值排除了图中特别大和特别小的点

- 取频率最高的F4共振峰的比值作为frequency参数

- 取时长的比值作为temporal参数

- 由于straight软件的限制，不能自由输入参数的值。下列的参数和理论上的结果有微小的不同。

  | 文件名           | F0     | Frequncy | Temporal |
  | ------------- | ------ | -------- | -------- |
  | sen6044AB.wav | 2.1734 | 1.1066   | 0.88291  |

#### `sen6147.wav`

|      | 基频                                       | 共振峰                                      | 基频最大值 | 基频最小值 | 共振峰1        | 共振峰2        | 共振峰3        | 共振峰4        | Time |
| ---- | ---------------------------------------- | ---------------------------------------- | ----- | ----- | ----------- | ----------- | ----------- | ----------- | ---- |
| A    | ![sen6147pitch](./data/实验2/A/sen6147pitch.png) | ![sen6147pitch](./data/实验2/A/sen6147formant.png) | 168.2 | 75.04 | 632.963884  | 1837.017269 | 3076.884396 | 4172.564597 | 7.04 |
| B    | ![sen6147pitch](./data/实验2/B/sen6147pitch.png) | ![sen6147pitch](./data/实验2/B/sen6147formant.png) | 398.6 | 85.10 | 1109.116500 | 2277.053848 | 3385.810323 | 4656.403556 | 6.75 |

- 取基频最大值和最小值平均值的比值作为F0参数,最大值和最小值排除了图中特别大和特别小的点

- 取频率最高的F4共振峰的比值作为frequency参数

- 取时长的比值作为temporal参数

- 由于straight软件的限制，不能自由输入参数的值。下列的参数和理论上的结果有微小的不同。

  | 文件名           | F0    | Frequncy | Temporal |
  | ------------- | ----- | -------- | -------- |
  | sen6147AB.wav | 2.035 | 1.1066   | 0.95499  |


### 结论

- 通过调节三个参数得到的新声音大体上是比较贴近目标的。可以通过直接对目标的基频，时长和共振峰的分析确定参数，一般不需要人反复辨别调节参数。
- 无法做到和目标声音完全相同。因为这些参数只能是对音频整体上做一些调节，没办法保证时时刻刻的细节都是一致的。

## 声音距离度量算法

### 实验任务

- 给出至少两种声音距离度量准则
- 对比实验1中的原始声音和转换后声音，利用你给出的声音距离度量准则，计算转换前后声音的距离
- 对实验2，利用你给出的声音距离度量准则，给出原始声音A到目标声音B的距离以及A到转换后声音A’的距离，说明为什么A’比A听上去更接近B的声音

### 思路

从上面的两个实验可以看出,声音的不同主要是由F0,Frequency和Temporal三个参数决定,所以度量距离也就从这三者参数出发.

使用straight工具包中的`exstraightsource`可以计算F0,使用`fft`可以计算频率,信号的长度就代表了Temporal

语音的能量和说话人没有什么太大的关系,所以这里不考虑音频的能量

### 度量算法一

#### 思路

为了度量Temporal的距离,将两个信号pad到相同的长度再进行后续的度量,这样就隐含地表示了Temporal.

为了度量F0的距离,使用`exstraightsource`计算F0,然后用某种距离算法度量F0的距离

为了度量Frequency的距离,使用`fft`计算Frequency,然后用某种距离算法度量Frequency的距离

最后将F0和Frequency综合考虑就得到了最终的距离.

而度量F0和Frequency的距离使用下面的公式:
$$
d = ({\sum_{i=1}^{n}|\frac{\vec{a}_i}{\sum_{i=1}^{n}\vec{a}_i}-\frac{\vec{b}_i}{\sum_{i=1}^{n}\vec{b}_i}|}) \cdot ({|\frac{1}{n}\sum_{i=1}^{n}\vec{a}_i-\frac{1}{n}\sum_{i=1}^{n}\vec{b}_i|})
$$
公式的前半部分将两个向量标准化,然后度量它们差向量的一范数,表示两个向量数值分布的相似程度.后半部分计算两个向量平均值的距离,表示两个向量中数值的大小.然后将两个参数相乘.



#### 算法描述

1. 首先将输入信号`a`, `b`向后补0到相同长度,这一步隐含了度量其Temporal.

   ``` matlab
   max_length = max(size(a, 1), size(b, 1));
   a = padarray(a, [max(max_length - size(a, 1), 0) 0], 'post');
   b = padarray(b, [max(max_length - size(b, 1), 0) 0], 'post');
   ```

2. 然后使用`exstraightsource`计算两个信号的`F0`矩阵

   ``` matlab
   f0a = exstraightsource(a, fsa);
   f0b = exstraightsource(b, fsb);
   ```

   ​

3. 使用`abs`, `fft`计算两个信号的频率(模)

   ``` matlab
   fa = abs(fft(a));
   fb = abs(fft(b));
   ```

4. 用下列公式计算F0矩阵的距离.公式的前一部分表示两者分布的相似程度,后一部分表示两者大小的相似程度

   ``` matlab
   f0_dis = sum(abs((f0a / sum(f0a) - f0b / sum(f0b)))) * abs(mean(f0a) - mean(f0b));
   ```

5.  用相同的公式计算两者的频率的距离.

   ``` matlab
   f_dis = sum(abs((fa / sum(fa) - fb / sum(fb)))) * abs(mean(fa) - mean(fb));
   ```

6. 将两个距离加权平均.系数是根据经验确定的.

   ``` matlab
   distance = f0_dis * 0.2 + f_dis * 0.8;
   ```

#### 结果分析

#### `guodegang.wav`

|      | `guodegang1.wav` | `guodegang2.wav` | `guodegang3.wav` |
| ---- | ---------------- | ---------------- | ---------------- |
| 距离   | 4.4957           | 10.2572          | 19.31            |

- 第一个的距离明显小于后两个,而在调节的时候第一个只改动了F0,改动也不大,听起来仍然是壮年男性的声音.而相比之下另外两个已经完全听不出来是男性的声音.所以这个结果是合理的.
- 而第二个和第三个相比,第二个声音的频率更低,原声音的相似程度确实更高.而距离算法给出的结果也反映出了这一点.

#### `shantianfang.wav`

|      | `shantianfang1.wav` | `shantianfang2.wav` | `shantianfang3.wav` |
| ---- | ------------------- | ------------------- | ------------------- |
| 距离   | 23.0864             | 27.4137             | 28.4363             |

#### `sen6000.wav`

|      | A       | A'     |
| ---- | ------- | ------ |
| 距离   | 13.5245 | 3.6546 |

#### `sen6015.wav`

|      | A       | A'    |
| ---- | ------- | ----- |
| 距离   | 15.1309 | 3.389 |

#### `sen6028.wav`

|      | A       | A'      |
| ---- | ------- | ------- |
| 距离   | 16.3738 | 11.4864 |

#### `sen6044.wav`

|      | A       | A'     |
| ---- | ------- | ------ |
| 距离   | 11.8689 | 4.0199 |

#### `sen6147.wav`

|      | A       | A'     |
| ---- | ------- | ------ |
| 距离   | 10.0325 | 9.3965 |

**A'和B的距离都明显比A小**

### 度量算法二

#### 思路

基本思路和前一个算法是相同的:

​	为了度量Temporal的距离,将两个信号pad到相同的长度再进行后续的度量,这样就隐含地表示了Temporal.

​	为了度量F0的距离,使用`exstraightsource`计算F0,然后用某种距离算法度量F0的距离

​	为了度量Frequency的距离,使用`fft`计算Frequency,然后用某种距离算法度量Frequency的距离

​	最后将F0和Frequency综合考虑就得到了最终的距离.

不同之处在于度量F0和Frequency使用了不同的方法:
$$
d=(1 - \sum_{i=1}^{n}\sqrt{\frac{\vec{a}_i}{\sum_{i=1}^{n}\vec{a}_i}\cdot\frac{\vec{b}_i}{\sum_{i=1}^{n}\vec{b}_i}}) \cdot \sqrt{||\vec{a}||_2||\cdot\vec{b}||_2}
$$
前一部分是将两个向量归一化之后,求其bhattacharyya距离,表示两个向量的数值分布的相似程度.后一部分是两个向量的模长.最后将两部分相乘.

#### 算法描述

1. 首先将输入信号`a`, `b`向后补0到相同长度,这一步隐含了度量其Temporal.

   ```matlab
   max_length = max(size(a, 1), size(b, 1));
   a = padarray(a, [max(max_length - size(a, 1), 0) 0], 'post');
   b = padarray(b, [max(max_length - size(b, 1), 0) 0], 'post');
   ```

2. 然后使用`exstraightsource`计算两个信号的`F0`矩阵

   ```matlab
   f0a = exstraightsource(a, fsa);
   f0b = exstraightsource(b, fsb);
   ```

3. 使用`abs`, `fft`计算两个信号的频率(模)

   ```matlab
   fa = abs(fft(a));
   fb = abs(fft(b));
   ```

4. 用下列公式计算F0矩阵的距离.公式的前一部分表示两者分布的相似程度,后一部分表示两者大小的相似程度

   ```matlab
   f0_dis = (1.0 - sum(sqrt(f0a / sum(f0a) .* f0b / sum(f0b)))) * sqrt(abs(dot(f0a, f0a) - dot(f0b, f0b)));
   ```

5. 用相同的公式计算两者的频率的距离.

   ```matlab
   f_dis = (1.0 - sum(sqrt(fa / sum(fa) .* fb / sum(fb)))) * sqrt(abs(dot(fa, fa) - dot(fb, fb)));
   ```

6. 将两个距离加权平均.系数是根据经验确定的.

   ```matlab
   distance = f0_dis * 0.4 + f_dis * 0.6;
   ```

#### 结果分析

#### `guodegang.wav`

|      | `guodegang1.wav` | `guodegang2.wav` | `guodegang3.wav` |
| ---- | ---------------- | ---------------- | ---------------- |
| 距离   | 1544.6919        | 3446.8489        | 9708.6752        |

这里可以得到和度量算法1相同的结论:

- 第一个的距离明显小于后两个,而在调节的时候第一个只改动了F0,改动也不大,听起来仍然是壮年男性的声音.而相比之下另外两个已经完全听不出来是男性的声音.所以这个结果是合理的.
- 而第二个和第三个相比,第二个声音的频率更低,原声音的相似程度确实更高.而距离算法给出的结果也反映出了这一点.

#### `shantianfang.wav`

|      | `shantianfang1.wav` | `shantianfang2.wav` | `shantianfang3.wav` |
| ---- | ------------------- | ------------------- | ------------------- |
| 距离   | 5096.0797           | 14137.545           | 8546.0523           |

#### `sen6000.wav`

|      | A         | A'        |
| ---- | --------- | --------- |
| 距离   | 2179.0456 | 1289.2465 |

#### `sen6015.wav`

|      | A         | A'       |
| ---- | --------- | -------- |
| 距离   | 1938.7322 | 730.0064 |

#### `sen6028.wav`

|      | A       | A'        |
| ---- | ------- | --------- |
| 距离   | 3164.29 | 1464.7978 |

#### `sen6044.wav`

|      | A         | A'       |
| ---- | --------- | -------- |
| 距离   | 1933.2546 | 556.0882 |

#### `sen6147.wav`

|      | A         | A'        |
| ---- | --------- | --------- |
| 距离   | 1406.5335 | 1278.9693 |

**A'和B的距离都明显比A小**

### 为什么A'更接近A

基频,频率和Temporal是我们感知声音的主要几个特征. 如果不去考虑一句话中每个音发音的细节,这几个参数就代表了这段音频整体上的音高和音色的特征.所以我们在实验二中才以使得这几个参数尽量接近B作为目标来生成A',实验三中的距离算法也基于这几个参数来考虑.从实验三的结果看来,和同一个的距离越小,听起来就是更相似.

### 代码运行说明

用MATLAB(r2017b)运行`run_lab3.m`即可.