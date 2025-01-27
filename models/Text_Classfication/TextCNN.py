import torch
import torch.nn.functional as F
from torch import nn

#https://blog.csdn.net/qq_28969139/article/details/103642946?spm=1001.2014.3001.5502
#https://zhuanlan.zhihu.com/p/339784219

class TextCNN(nn.Module):
 
    def __init__(self):
        super(TextCNN, self).__init__()
        num_embeddings = 95811
        num_classes = 4
 
        embedding_dim = 300  # 300
        num_kernel = 100  # 100
        kernel_sizes = [2,3,4]  # 3,4,5
        dropout = 0.5  # 0.5会导致测试集比训练集效果好很多。
 
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_kernel, (k, embedding_dim)) for k in kernel_sizes])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_kernel * len(kernel_sizes), num_classes, bias=True)
        )
 
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [conv(x).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x

def textcnn():
    return TextCNN()