# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Ref: https://cxyzjd.com/article/kingsonyoung/90757879#41FastText_68

class FastText(nn.Module):
    def __init__(self):
        num_embeddings = 95811
        vec_dim = 300
        label_size = 4
        hidden_size = 256
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(num_embeddings, vec_dim)
        #self.embed.weight.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, label_size)
        )

    def forward(self, x):
        x = self.embed(x)
        out = self.fc(torch.mean(x, dim=1))
        return out

def fasttext():
    return FastText()