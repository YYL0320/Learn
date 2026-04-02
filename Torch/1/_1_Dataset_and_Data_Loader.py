# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
#
# '''
# 使用以下参数加载 FashionMNIST 数据集
# root 是存储训练/测试数据的路径，
# train 指定训练集或测试集，
# download=True 如果数据在 root 目录下不可用，则从互联网下载数据。
# transform 和 target_transform 指定特征和标签的转换
# '''
# training_data = datasets.FashionMNIST(
#     root=r"D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\FashionMNIST\train",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root=r"D:\ranjaian\pychram\PyCharm 2025.2.3\PythonProject1\Torch\data\FashionMNIST\test",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
# #可以像列表一样手动索引 Datasets：training_data[index]。我们使用 matplotlib 可视化训练数据中的一些样本。
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

#自定义数据集
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#使用 DataLoaders 为训练准备数据
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#迭代 DataLoader
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")