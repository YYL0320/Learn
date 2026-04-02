import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#获取训练设备
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#定义类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#取一个包含 3 个 28x28 图像的样本小批量
input_image = torch.rand(3,28,28)
print(input_image.size())

#初始化 nn.Flatten 层
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#使用其存储的权重和偏差对输入应用线性变换（线性层）
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#非线性激活（非线性层）
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#一个有序的模块容器
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#最后一个线性层返回logits，logits 被缩放到 [0, 1] 范围内的值，dim 参数指示了必须将值求和为 1 的维度
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

#模型参数
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")