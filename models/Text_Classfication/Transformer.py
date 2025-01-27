import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#要多跑几次迭代

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=8, num_encoder_layers=6, dim_feedforward=256, dropout=0.5):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
    #     self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.fc_out.bias.data.zero_()
    #     self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(len(src))
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.fc_out(output)
        return F.log_softmax(output, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Example usage
def transformer():
    return Transformer(vocab_size=95811, num_classes=4)
