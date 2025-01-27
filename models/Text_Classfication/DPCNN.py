# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''
# Ref: https://github.com/Cheneng/DPCNN

class DPCNN(nn.Module):
    def __init__(self):
        super(DPCNN, self).__init__()
        
        self.channel_size = 250
        self.embedding = nn.Embedding(95811, 300)

        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, 300), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, 4)

    def forward(self, x):
        batch = x.shape[0]

        # Region embedding
        x = self.embedding(x)  # Convert word indices to embeddings
        #print(x.shape)
        x = x.unsqueeze(1)
        #print(x.shape)
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]
        #print(x.shape)
        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        #print(x.shape)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        
        while x.size()[-2] > 2:
            x = self._block(x)
        #print(x.shape)  # 打印当前的x形状

        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = x.view(batch, self.channel_size)
        #x = x.view(batch, -1)
        #print(x.shape)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)   #最后倒数第二维的奇偶会导致一会输出1，一会输出0.
        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        # Short Cut
        x = x + px

        return x
    
def dpcnn():
    return DPCNN()

