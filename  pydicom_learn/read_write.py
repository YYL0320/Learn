from pydicom import examples, dcmread

# path = examples.get_path("ct")
# print(path)
# with open(path, 'rb') as infile:
#     ds = dcmread(infile)
#     print(ds)
# with dcmread(path) as ds:
#    print(type(ds))
# no_meta_path = examples.get_path('no_meta')
# # ds = dcmread(no_meta_path)
# ds = dcmread(no_meta_path, force=True)
# print(ds)
#
# from pydicom import examples
# ds = examples.ct
# type(ds)

# ds=dcmread(r"D:\wenjian\LIDC-IDRI\raw\LIDC-IDRI\LIDC-IDRI-0132\01-01-2000-NA-NA-67346\5418.000000-ThorRoutine  3.0  B31f-62989\1-001.dcm")
# print(ds)
#
# print("————————————————————————————————————————————————————————————————————")
#
# print(ds[0x0010,0x0020])
# print(ds.PatientID)
# print(len(ds.PatientID))
# print(ds["PatientID"])
#
# print(ds.ImageType)
# print(ds.ImageType[1])
#
# print("————————————————————————————————————————————————————————————————————")
#
# print("文件元数据\n",ds.file_meta)
#
# print("可直接用=来修改元素值")
# print("可修改可删除")
#
# print('WindowWidth' in ds)
#
# print("使用ds.save_as('新名称')来保存新的dcm文件")
#
# print("可以写入任何形式的python文件")
# from io import BytesIO
# out = BytesIO()
# ds.save_as(out)
# print("保存前需要添加UID元素")
# ds.save_as('out.dcm', enforce_file_format=True)



ds = examples.jpeg2k
print(ds.group_dataset(0x7FE0))

tsyntax = ds.file_meta.TransferSyntaxUID
print("压缩方式",tsyntax.name)

ds = examples.ct
tsyntax = ds.file_meta.TransferSyntaxUID
print(tsyntax.name)
print('是否压缩：',tsyntax.is_compressed)