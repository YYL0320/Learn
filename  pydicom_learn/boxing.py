from pydicom import examples  # 导入pydicom库中的examples模块，用于获取pydicom内置的示例DICOM文件

# 获取一个波形(Waveform)示例数据集
ds = examples.waveform

# 打印SOPClassUID的名称属性，SOPClassUID标识该DICOM文件的类型（如12导联心电图存储）
print(ds.SOPClassUID.name)
# 输出: '12-lead ECG Waveform Storage'，表示这是一个12导联心电图波形存储

# 从数据集中提取波形序列(WaveformSequence)
waveforms = ds.WaveformSequence

# 打印波形序列的长度，即包含的波形数量
print(len(waveforms))