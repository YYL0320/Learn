import torch
#
# class TinyModel(torch.nn.Module):
#
#     def __init__(self):
#         super(TinyModel, self).__init__()
#
#         self.linear1 = torch.nn.Linear(100, 200)
#         self.activation = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(200, 10)
#         self.softmax = torch.nn.Softmax()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         x = self.softmax(x)
#         return x
#
# tinymodel = TinyModel()
#
# print('The model:')
# print(tinymodel)
#
# print('\n\nJust one layer:')
# print(tinymodel.linear2)
#
# print('\n\nModel params:')
# for param in tinymodel.parameters():
#     print(param.numel())
#
# print('\n\nLayer1 params:')
# for param in tinymodel.linear1.parameters():
#     print(param.numel())
#
# print('\n\nLayer2 params:')
# for param in tinymodel.linear2.parameters():
#     print(param.numel())

import torch.nn.functional as F

#卷积层
class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features