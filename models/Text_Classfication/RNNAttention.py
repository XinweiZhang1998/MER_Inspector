# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''



class RNNAttention(nn.Module):
    def __init__(self):
        super(RNNAttention, self).__init__()
        vocab_size = 95811          # 词表大小
        embedding_dim = 300        # 词向量维度
        hidden_size = 128
        hidden_size2 = 64
        num_layers = 2
        dropout = 0.5
        num_classes = 4
        batch_size = 32  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        # self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        #print(emb.shape)
        lstmout,(c,h) = self.lstm(emb)
        #print (lstmout.shape)
        M = self.tanh1(lstmout)
        #print (M.shape)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = lstmout * alpha
        out = torch.sum(out, axis=1)

        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def rnnattention():
    return RNNAttention()