from pydicom import examples
ds = examples.waveform
print(ds.SOPClassUID.name)
'12-lead ECG Waveform Storage'
waveforms = ds.WaveformSequence
print(len(waveforms))
