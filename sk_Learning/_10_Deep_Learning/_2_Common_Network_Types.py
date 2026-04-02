# FCN
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFCN, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, 128)  # 第一隐藏层
        self.relu = nn.ReLU()                  # 激活函数
        self.fc2 = nn.Linear(128, 64)         # 第二隐藏层
        self.fc3 = nn.Linear(64, num_classes) # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 假设输入是100维的特征，进行10分类
model = SimpleFCN(input_size=100, num_classes=10)

#RNN
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) # 词嵌入层
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x 的形状: (batch_size, sequence_length)
        x = self.embedding(x) # 嵌入后: (batch_size, seq_len, embed_size)
        _, h_n = self.rnn(x)  # h_n 是最后一个时间步的隐藏状态
        out = self.fc(h_n.squeeze(0)) # 用最后的状态进行分类
        return out

#GAN
# GAN的核心训练循环伪代码示意
for epoch in range(num_epochs):
    # 1. 训练判别器：最大化判别真实数据为真、生成数据为假的能力
    real_data = get_real_data()
    noise = generate_random_noise()
    fake_data = generator(noise).detach() # 注意detach，防止生成器被更新

    d_loss_real = criterion(discriminator(real_data), real_labels)
    d_loss_fake = criterion(discriminator(fake_data), fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()

    # 2. 训练生成器：最小化判别器将生成数据判为假的能力（即骗过判别器）
    noise = generate_random_noise()
    fake_data = generator(noise)
    g_loss = criterion(discriminator(fake_data), real_labels) # 让判别器认为生成的是真的
    g_loss.backward()
    optimizer_G.step()