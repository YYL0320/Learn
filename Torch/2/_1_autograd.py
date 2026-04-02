# import torch
#
# if torch.accelerator.is_available():
#     print('We have an accelerator!')
# else:
#     print('Sorry, CPU only.')

# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

# a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
# print(a)
#
# b = torch.sin(a)
# plt.plot(a.detach(), b.detach())
# plt.show()
#
# c = 2 * b
# print(c)
#
# d = c + 1
# print(d)
#
# out = d.sum()
# print(out)
#
# out.backward()
# print(a.grad)
# plt.plot(a.detach(), a.grad.detach())
# plt.show()


#一个小模型
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()

#尚未计算任何梯度
print(model.layer2.weight[0][0:10]) # just a small slice
print(model.layer2.weight.grad)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
print(loss)

#调用 loss.backward()
loss.backward()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])

#运行优化器。优化器负责根据计算出的梯度更新模型权重。
optimizer.step()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])