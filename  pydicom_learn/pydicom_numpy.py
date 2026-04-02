# import matplotlib.pyplot as plt
#
# from pydicom import dcmread
# from pydicom.pixels import pixel_array
#
# # Get an example dataset as a FileDataset instance
# ds = r"D:\wenjian\LIDC-IDRI\raw\LIDC-IDRI\LIDC-IDRI-0132\01-01-2000-NA-NA-67346\5418.000000-ThorRoutine  3.0  B31f-62989\1-001.dcm"
# ds = dcmread(ds)
#
# # print(ds)
# # Convert the pixel data to an ndarray
# arr = pixel_array(ds)
# # assert arr.shape == (512,512)       #判断语句，只有两边相等是才会继续执行
# # print(str(arr.dtype))
# # assert str(arr.dtype) == "uint16"
#
# # Display the pixel data using matplotlib
# plt.imshow(arr, cmap="gray")
# plt.show()


import matplotlib.pyplot as plt

# from pydicom import examples,dcmread
# from pydicom.pixels import pixel_array
#
# ds = r"D:\wenjian\LIDC-IDRI\raw\LIDC-IDRI\LIDC-IDRI-0132\01-01-2000-NA-NA-67346\5418.000000-ThorRoutine  3.0  B31f-62989\1-001.dcm"
# ds = dcmread(ds)
# # ds = examples.ybr_color
# # assert ds.PhotometricInterpretation == "YBR_FULL_422"
# # print(ds.PhotometricInterpretation)
#
# # ybr = pixel_array(ds, index=0, raw=True)    # index=0：取第0帧（单帧影像）；raw=True：不做任何格式转换，返回原始数据     #不需要格式转换
# rgb = pixel_array(ds, index=0)      #省略raw=True，默认自动转换为RGB
# hu_array = rgb * ds.RescaleSlope + ds.RescaleIntercept      #转换为临床常用的HU值
# fig, (hu_ax1) = plt.subplots(nrows=1)
# # fig, (im1, im2) = plt.subplots(1, 2)
# # im1.imshow(ybr)
# # im1.set_title("Original (in YCbCr)")
# hu_ax1.imshow(rgb)
# hu_ax1.set_title("Converted (in RGB)")
# plt.show()

# 4. 可视化（针对MONOCHROME2的灰度显示，设置不同窗宽窗位）
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#
# # 肺窗（-1000 ~ 400 HU）：看肺部结构
# ax1.imshow(hu_array, cmap="gray", vmin=-1000, vmax=400)
# ax1.set_title("Lung Window (MONOCHROME2)")
# ax1.axis("off")
#
# # 纵隔窗（-150 ~ 250 HU）：看纵隔/淋巴结
# ax2.imshow(hu_array, cmap="gray", vmin=-150, vmax=250)
# ax2.set_title("Mediastinal Window")
# ax2.axis("off")
#
# # 骨窗（200 ~ 1500 HU）：看骨骼
# ax3.imshow(hu_array, cmap="gray", vmin=200, vmax=1500)
# ax3.set_title("Bone Window")
# ax3.axis("off")
#
# plt.tight_layout()
# plt.show()


# from pydicom import examples
# from pydicom.pixels import pixel_array
#
# # ds = examples.jpeg2k
# #
# # # Returns the results from the first successful decoding plugin
# # arr = pixel_array(ds)
# #
# fig, x1 = plt.subplots()
#
# # Get the path to the 'examples.rt_dose' dataset
# path = examples.get_path("rt_dose")
#
# # Return the first frame of the pixel data
# arr = pixel_array(path, index=0)
# x1.imshow(arr)
# plt.show()

# from pydicom import Dataset, examples
# from pydicom.pixels import pixel_array, apply_rescale
# import matplotlib.pyplot as plt
#
# # 1. 获取示例CT文件路径并读取
# path = examples.get_path("ct")
# ds = Dataset()
# arr = pixel_array(path, ds_out=ds)  # 读取像素数组 + 填充DICOM标签到ds
#
# # 3. 转换为HU值（两种方式对比，结果一致）
# # 方式1：官方函数 apply_rescale（推荐，更健壮）
# hu = apply_rescale(arr, ds)
# # 方式2：手动计算（等价，验证逻辑）
# hu_manual = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
#
# fig, (x1,x2) = plt.subplots(1,2)
# # 4. 可视化HU值（CT肺窗）
# x1.imshow(hu_manual,cmap="gray")  # 肺窗范围
# x2.imshow(hu,cmap="gray")
# plt.show()


import matplotlib.pyplot as plt

from pydicom import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

import numpy as np

def draw_circle(shape: tuple[int, int], dtype: str, value: int) -> np.ndarray:
    """Return an ndarray containing a circle."""
    (rows, columns), radius = shape, min(shape) // 2

    x0, y0 = columns // 2, rows // 2
    x = np.linspace(0, columns, columns)
    y = np.linspace(0, rows, rows)[:, None]

    # Create a boolean array where values inside the radius are True
    arr = (x - x0)**2 + (y - y0)**2 <= radius**2

    # Convert to the required `dtype` and set the maximum `value`
    return arr.astype(dtype) * value

ds = Dataset()
ds.file_meta = FileMetaDataset()
ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

ds.BitsAllocated = 8  # 8-bit containers
ds.BitsStored = 8  # 8-bits used
ds.HighBit = ds.BitsStored - 1
ds.PixelRepresentation = 0  # unsigned

ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"

## Even number of bytes
# Create a 480 x 320, 8-bit unsigned array
arr = draw_circle((320, 480), "uint8", 255)
assert arr.size % 2 == 0

# No padding needed
ds.PixelData = arr.tobytes()
ds["PixelData"].VR = "OB"
ds.Rows = arr.shape[0]  # 320 pixels
ds.Columns = arr.shape[1]  # 480 pixels

plt.imshow(ds.pixel_array)
plt.show()

## Odd number of bytes
# Create a 31 x 63, 8-bit unsigned array
arr = draw_circle((63, 31), "uint8", 255)
assert arr.size % 2 == 1

# Trailing padding required to make the length an even number of bytes
ds.PixelData = b"".join((arr.tobytes(), b"\x00"))
ds["PixelData"].VR = "OB"
ds.Rows = arr.shape[0]
ds.Columns = arr.shape[1]

plt.imshow(ds.pixel_array)
plt.show()