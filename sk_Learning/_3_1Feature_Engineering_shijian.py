from seaborn import load_dataset

df=load_dataset('titanic')

print('前五行数据')
print(df.head())

print(df.info())

print(df.isnull().sum())

