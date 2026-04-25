# =============================================================================
# 以下代码展示如何使用matplotlib和pydicom读取DICOM文件并显示像素数据
# =============================================================================

# 导入matplotlib的pyplot模块用于数据可视化
# import matplotlib.pyplot as plt

# 从pydicom库导入dcmread函数，用于读取DICOM文件
# from pydicom import dcmread

# 从pydicom.pixels模块导入pixel_array函数，用于将DICOM像素数据转换为numpy数组
# from pydicom.pixels import pixel_array

# 方法1：使用dcmread直接读取DICOM文件路径（返回一个FileDataset对象）
# ds = r"D:\wenjian\LIDC-IDRI\raw\LIDC-IDRI\LIDC-IDRI-0132\01-01-2000-NA-NA-67346\5418.000000-ThorRoutine  3.0  B31f-62989\1-001.dcm"
# ds = dcmread(ds)  # 读取指定路径的DICOM文件

# 打印整个DICOM数据集的详细信息（包含所有标签和值）
# print(ds)

# 方法2：使用pixel_array函数将像素数据转换为numpy数组
# arr = pixel_array(ds)

# assert语句用于调试时的断言检查，如果条件不满足则抛出AssertionError
# assert arr.shape == (512,512)  # 验证数组形状是否为512x512

# 打印数组的数据类型（如uint16表示无符号16位整数）
# print(str(arr.dtype))

# 验证数据类型是否为uint16
# assert str(arr.dtype) == "uint16"

# 使用matplotlib显示像素数据，cmap="gray"表示使用灰度色彩映射
# plt.imshow(arr, cmap="gray")
# plt.show()  # 显示图像窗口


# =============================================================================
# 以下代码展示如何读取DICOM文件并进行HU值转换和RGB显示
# =============================================================================

import matplotlib.pyplot as plt  # 导入matplotlib用于图像显示

# 导入pydicom的examples模块（获取示例数据）和dcmread函数（读取DICOM文件）
# from pydicom import examples, dcmread

# 导入pixel_array函数用于提取像素数据
# from pydicom.pixels import pixel_array

# 指定一个真实的DICOM文件路径（CT扫描图像）
# ds = r"D:\wenjian\LIDC-IDRI\raw\LIDC-IDRI\LIDC-IDRI-0132\01-01-2000-NA-NA-67346\5418.000000-ThorRoutine  3.0  B31f-62989\1-001.dcm"
# ds = dcmread(ds)  # 读取DICOM文件

# 示例：使用pydicom内置的ybr_color示例数据
# ds = examples.ybr_color  # YBR颜色空间的示例数据

# 验证图像的 photometric interpretation（光度解释）是否为YBR_FULL_422
# 这是YUV色彩空间的一种，用于JPEG压缩的DICOM图像
# assert ds.PhotometricInterpretation == "YBR_FULL_422"
# print(ds.PhotometricInterpretation)  # 打印光度解释值

# 使用pixel_array提取像素数据：
# index=0：取第0帧数据（对于多帧图像有效，单帧图像此参数无影响）
# raw=True：返回原始数据，不进行任何色彩空间转换
# ybr = pixel_array(ds, index=0, raw=True)

# 省略raw=True参数，默认自动将YBR色彩空间转换为RGB
# rgb = pixel_array(ds, index=0)

# HU值转换公式：HU = pixel_value * RescaleSlope + RescaleIntercept
# 这是将DICOM原始像素值转换为临床常用的Hounsfield Unit（亨氏单位）的方法
# HU值用于CT图像，-1000表示空气，0表示水
# hu_array = rgb * ds.RescaleSlope + ds.RescaleIntercept

# 创建图像窗口，包含一个子图
# fig, (hu_ax1) = plt.subplots(nrows=1)

# 如果需要创建两个子图对比YBR和RGB：
# fig, (im1, im2) = plt.subplots(1, 2)

# 显示原始YBR图像（未转换色彩空间）
# im1.imshow(ybr)
# im1.set_title("Original (in YCbCr)")

# 显示RGB图像
# hu_ax1.imshow(rgb)
# hu_ax1.set_title("Converted (in RGB)")
# plt.show()


# =============================================================================
# 以下代码展示如何设置不同的窗宽窗位来可视化CT图像
# 窗宽窗位用于调整CT图像的对比度和亮度，以便观察不同组织
# =============================================================================

# 4. 可视化（针对MONOCHROME2的灰度显示，设置不同窗宽窗位）

# 创建包含3个子图的图像窗口， figsize设置窗口大小为15x5英寸
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 肺窗（-1000 ~ 400 HU）：主要观察肺部结构
# vmin和vmax设置显示的灰度范围，范围外的值分别显示为全黑和全白
# ax1.imshow(hu_array, cmap="gray", vmin=-1000, vmax=400)
# ax1.set_title("Lung Window (MONOCHROME2)")
# ax1.axis("off")  # 隐藏坐标轴

# 纵隔窗（-150 ~ 250 HU）：主要观察纵隔和淋巴结
# ax2.imshow(hu_array, cmap="gray", vmin=-150, vmax=250)
# ax2.set_title("Mediastinal Window")
# ax2.axis("off")

# 骨窗（200 ~ 1500 HU）：主要观察骨骼结构
# ax3.imshow(hu_array, cmap="gray", vmin=200, vmax=1500)
# ax3.set_title("Bone Window")
# ax3.axis("off")

# 自动调整子图布局，防止重叠
# plt.tight_layout()
# plt.show()


# =============================================================================
# 以下代码展示如何读取RT DOSE（放疗剂量）文件并显示
# =============================================================================

# from pydicom import examples  # 导入pydicom示例模块
# from pydicom.pixels import pixel_array  # 导入像素数组提取函数

# 获取pydicom内置的jpeg2k示例数据
# ds = examples.jpeg2k

# pixel_array函数会尝试使用第一个成功的解码插件来解码像素数据
# arr = pixel_array(ds)

# 创建图像窗口和一个子图
# fig, x1 = plt.subplots()

# 获取rt_dose示例文件的路径
# path = examples.get_path("rt_dose")

# 返回像素数据的第一帧（index=0）
# RT Dose是多帧图像，可能包含多个层面的剂量信息
# arr = pixel_array(path, index=0)
# x1.imshow(arr)
# plt.show()


# =============================================================================
# 以下代码展示如何使用apply_rescale函数和手动计算两种方式转换HU值
# =============================================================================

# from pydicom import Dataset, examples  # 导入Dataset类和一些示例数据
# from pydicom.pixels import pixel_array, apply_rescale  # 导入像素处理相关函数
# import matplotlib.pyplot as plt  # 导入绘图模块

# 1. 获取示例CT文件路径并读取
# path = examples.get_path("ct")  # 获取CT示例文件路径

# 创建空的Dataset对象
# ds = Dataset()

# pixel_array函数读取像素数组，并通过ds_out参数将DICOM标签填充到ds对象中
# 这样ds对象就包含了完整的DICOM信息
# arr = pixel_array(path, ds_out=ds)

# 3. 转换为HU值（两种方式对比，结果一致）

# 方式1：使用官方提供的apply_rescale函数（推荐，更健壮）
# apply_rescale函数内部处理了数据类型转换和偏移量计算
# hu = apply_rescale(arr, ds)

# 方式2：手动计算HU值
# HU = pixel_value * RescaleSlope + RescaleIntercept
# 需要将RescaleSlope和RescaleIntercept转换为float类型
# hu_manual = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

# 创建包含两个子图的图像窗口
# fig, (x1,x2) = plt.subplots(1,2)

# 4. 可视化HU值（使用CT肺窗参数）
# 肺窗范围：vmin=-1000, vmax=400，适合观察肺组织
# x1.imshow(hu_manual,cmap="gray")  # 显示手动计算的HU值
# x2.imshow(hu,cmap="gray")  # 显示apply_rescale计算的HU值
# plt.show()


# =============================================================================
# 以下是实际运行的代码：创建一个简单的圆形图像并保存为DICOM格式
# =============================================================================

# 导入matplotlib的pyplot模块用于图像显示
import matplotlib.pyplot as plt

# 从pydicom导入Dataset和FileMetaDataset类
# Dataset：表示DICOM数据集，包含像素数据和元数据
# FileMetaDataset：表示DICOM文件的元数据信息（不影响像素数据显示）
from pydicom import Dataset, FileMetaDataset

# 导入传输语法UID相关的明确VR小端字节序传输语法
# ExplicitVRLittleEndian：显式VR（Value Representation）小端字节序
# 这是DICOM标准中常用的传输语法之一
from pydicom.uid import ExplicitVRLitEndian

import numpy as np  # 导入numpy库用于数值计算


def draw_circle(shape: tuple[int, int], dtype: str, value: int) -> np.ndarray:
    """
    创建一个包含圆形的numpy数组。

    参数:
        shape: 元组 (行数, 列数)，指定输出数组的形状
        dtype: 字符串，指定输出数组的数据类型（如"uint8", "uint16"）
        value: 整数，圆形内部填充的像素值

    返回:
        np.ndarray：一个二维数组，圆形内部为value，外部为0
    """
    # 使用元组解包获取行数和列数
    (rows, columns), radius = shape, min(shape) // 2

    # 计算圆心坐标（使用整数除法）
    x0, y0 = columns // 2, rows // 2

    # 使用numpy的linspace生成坐标数组
    # x是列坐标，y是行坐标（使用[:, None]将其转换为列向量）
    x = np.linspace(0, columns, columns)
    y = np.linspace(0, rows, rows)[:, None]

    # 计算每个点到圆心的距离平方，判断是否在圆内
    # 距离平方 <= 半径平方 的点被标记为True（表示在圆内）
    arr = (x - x0)**2 + (y - y0)**2 <= radius**2

    # 将布尔数组转换为指定的dtype，并乘以value
    # True * value = value, False * value = 0
    return arr.astype(dtype) * value


# 创建空的Dataset对象
ds = Dataset()

# 创建空的FileMetaDataset对象，用于存储DICOM文件元信息
ds.file_meta = FileMetaDataset()

# 设置传输语法UID为显式VR小端字节序
# 这是DICOM标准的默认传输语法之一，兼容性最好
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

# 设置像素数据相关的DICOM标签
ds.BitsAllocated = 8  # 每像素分配的位数，这里是8位（1字节）
ds.BitsStored = 8  # 每像素实际存储的位数
ds.HighBit = ds.BitsStored - 1  # 最高有效位索引
ds.PixelRepresentation = 0  # 像素表示方式：0=无符号整数，1=有符号整数

# 色彩相关设置
ds.SamplesPerPixel = 1  # 每个像素的样本数（1表示灰度图，3表示彩色图）
ds.PhotometricInterpretation = "MONOCHROME2"  # 光度解释：MONOCHROME2=灰度（白=亮）

## ========== 示例1：偶数字节数（不需要填充）==========

# 创建一个320行480列的8位无符号数组，像素值255（白色）
arr = draw_circle((320, 480), "uint8", 255)

# 断言验证数组大小为偶数字节（某些传输语法要求）
assert arr.size % 2 == 0

# 将数组转换为字节流并赋值给PixelData
ds.PixelData = arr.tobytes()

# 手动设置PixelData的VR（Value Representation）为"OB"（Other Byte）
# 这是像素数据的标准VR类型
ds["PixelData"].VR = "OB"

# 设置图像的行列数
ds.Rows = arr.shape[0]  # 320像素（行数）
ds.Columns = arr.shape[1]  # 480像素（列数）

# 使用matplotlib显示DICOM图像
plt.imshow(ds.pixel_array)
plt.show()

## ========== 示例2：奇数字节数（需要填充）==========

# 创建一个63行31列的8位无符号数组
arr = draw_circle((63, 31), "uint8", 255)

# 断言验证数组大小为奇数字节
assert arr.size % 2 == 1

# DICOM标准要求像素数据长度为偶数字节
# 因此需要添加一个字节的填充（trailing padding）
# 使用b"".join()将原始字节和填充字节\x00拼接
ds.PixelData = b"".join((arr.tobytes(), b"\x00"))

# 手动设置PixelData的VR为"OB"
ds["PixelData"].VR = "OB"

# 设置图像的行列数
ds.Rows = arr.shape[0]  # 63像素（行数）
ds.Columns = arr.shape[1]  # 31像素（列数）

# 使用matplotlib显示DICOM图像
plt.imshow(ds.pixel_array)
plt.show()