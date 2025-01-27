import torch
import torch.nn.functional as F
from torch import nn
 
#Learning Rate =0.1
# Ref：https://blog.csdn.net/qq_28969139/article/details/103646076
 
class VDCNN(nn.Module):
 
    def __init__(self, num_layers):
        super(VDCNN, self).__init__()
        num_embeddings = 95811
        num_classes = 4
        num_layers = num_layers
        layers_types = {
            9: [2, 2, 2, 2], 
            17: [4, 4, 4, 4],
            29: [4, 4, 10, 10],
            49: [6, 10, 16, 16]
        }
        layers_dist = layers_types[num_layers]
 
        self.embed = nn.Embedding(num_embeddings, 300, 0) #这个embed词为16维，dpcnn为300.这个影响大吗？
        self.conv = nn.Conv1d(300, 64, 3, 1, 1)
        self.conv_block1 = nn.Sequential(
            *([ConvBlock(64, 64, 3)] + [ConvBlock(64, 64, 3) for _ in range(layers_dist[0] - 1)]))
 
        self.conv_block2 = nn.Sequential(
            *([ConvBlock(64, 128, 3)] + [ConvBlock(128, 128, 3) for _ in range(layers_dist[1] - 1)]))
 
        self.conv_block3 = nn.Sequential(
            *([ConvBlock(128, 256, 3)] + [ConvBlock(256, 256, 3) for _ in range(layers_dist[2] - 1)]))
 
        self.conv_block4 = nn.Sequential(
            *([ConvBlock(256, 512, 3)] + [ConvBlock(512, 512, 3) for _ in range(layers_dist[3] - 1)]))
 
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
 
    # input_length=1024
    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)
        x = self.conv_block1(x)
        x = F.max_pool1d(x, 3, 2, 1)
        x = self.conv_block2(x)
        x = F.max_pool1d(x, 3, 2, 1)
        x = self.conv_block3(x)
        x = F.max_pool1d(x, 3, 2, 1)
        x = self.conv_block4(x)
        x, _ = x.topk(8, dim=2, sorted=False)
        x = x.view(x.size(0), -1).contiguous()
        x = self.fc(x)
        return x
 
 
class ConvBlock(nn.Module):
 
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x = self.shortcut(x)
        return y + x


def vdcnn_9():
    return VDCNN(num_layers=9)

def vdcnn_17():
    return VDCNN(num_layers=17)

def vdcnn_29():
    return VDCNN(num_layers=29)

def vdcnn_49():
    return VDCNN(num_layers=49)