from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# 统计可训练参数总数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数数量: {trainable_params}")  # 应该输出 5130 (512*10 + 10偏置)