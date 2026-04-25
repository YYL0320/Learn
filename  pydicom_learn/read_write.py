# =============================================================================
# 以下代码展示pydicom读取DICOM文件的各种方式
# =============================================================================

# 从pydicom库导入examples模块（包含示例数据）和dcmread函数（用于读取DICOM文件）
from pydicom import examples, dcmread

# 使用examples.get_path()获取pydicom内置CT示例文件的路径
# path = examples.get_path("ct")
# print(path)  # 打印文件路径

# 方法1：使用with语句和dcmread读取DICOM文件（自动管理文件资源）
# with open(path, 'rb') as infile:  # 以二进制读取模式打开文件
#     ds = dcmread(infile)  # 使用dcmread解析DICOM文件流
#     print(ds)  # 打印整个数据集

# 方法2：直接使用dcmread读取文件路径（更简洁的写法）
# with dcmread(path) as ds:
#     print(type(ds))  # 打印数据类型，通常是FileDataset

# 获取一个没有文件元信息的示例数据集
# no_meta_path = examples.get_path('no_meta')

# 读取时不带force=True会使用默认行为
# ds = dcmread(no_meta_path)

# 使用force=True强制读取，即使文件缺少必要的元信息
# ds = dcmread(no_meta_path, force=True)
# print(ds)

# 直接访问pydicom内置的示例数据集（ct是一个完整的Dataset对象）
# ds = examples.ct
# type(ds)  # 返回pydicom.dataset.Dataset类型


# =============================================================================
# 以下代码展示如何读取用户指定的DICOM文件并访问其标签
# =============================================================================

# 使用dcmread读取用户指定的DICOM文件路径
# ds = dcmread(r"D:\wenjian\LIDC-IDRI\raw\LIDC-IDRI\LIDC-IDRI-0132\01-01-2000-NA-NA-67346\5418.000000-ThorRoutine  3.0  B31f-62989\1-001.dcm")
# print(ds)  # 打印完整的DICOM数据集

# 打印分隔线，便于区分不同部分的输出
# print("————————————————————————————————————————————————————————————————————")

# 方法1：使用标签的Group和Element编号访问（十六进制）
# DICOM标签由(Group, Element)组成，如(0010,0020)是患者ID
# print(ds[0x0010,0x0020])

# 方法2：直接使用属性名访问（更直观）
# print(ds.PatientID)  # 患者ID

# 打印患者ID的长度
# print(len(ds.PatientID))

# 方法3：使用字典方式访问标签
# print(ds["PatientID"])

# 打印ImageType标签（一个字符串列表）
# print(ds.ImageType)
# print(ds.ImageType[1])  # 打印ImageType列表中的第二个元素

# 打印分隔线
# print("————————————————————————————————————————————————————————————————————")


# =============================================================================
# 以下代码展示DICOM文件的元数据操作
# =============================================================================

# 打印文件元数据（包含传输语法、媒体存储SOP类等重要信息）
# print("文件元数据\n",ds.file_meta)

# 直接使用等号赋值修改元素值
# print("可直接用=来修改元素值")

# 使用in操作符检查标签是否存在
# print('WindowWidth' in ds)  # 返回True或False

# 使用save_as方法保存DICOM文件
# print("使用ds.save_as('新名称')来保存新的dcm文件")

# save_as方法可以将DICOM数据保存为任何格式的Python文件
# from io import BytesIO  # 导入BytesIO用于内存中操作二进制数据
# out = BytesIO()
# ds.save_as(out)  # 将数据集保存到BytesIO对象中
# print("保存前需要添加UID元素")  # 某些格式需要特定的UID
# ds.save_as('out.dcm', enforce_file_format=True)  # 强制保存为DICOM格式


# =============================================================================
# 以下代码展示如何检查DICOM文件的压缩状态和传输语法
# =============================================================================

# 获取jpeg2k示例数据集
ds = examples.jpeg2k

# 获取像素数据组（0x7FE0）的数据集，主要包含PixelData标签
# print(ds.group_dataset(0x7FE0))

# 获取文件元数据中的传输语法UID
tsyntax = ds.file_meta.TransferSyntaxUID

# 打印传输语法名称，如"JPEG 2000 Image Compression"
print("压缩方式", tsyntax.name)

# 获取CT示例数据集
ds = examples.ct

# 获取CT文件的传输语法UID
tsyntax = ds.file_meta.TransferSyntaxUID

# 打印传输语法名称
print(tsyntax.name)

# 判断传输语法是否为压缩格式
# is_compressed属性返回True表示数据是压缩的，False表示未压缩
print('是否压缩：', tsyntax.is_compressed)