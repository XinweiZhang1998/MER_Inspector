#Hierarchical Attention Network (HAN) 
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义单词级别的注意力网络
class WordAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, dropout):
        super(WordAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size * max_sentence_number, max_sentence_length, embed_dim]
        x = self.dropout(x)
        gru_out, _ = self.gru(x)  # [batch_size * max_sentence_number, max_sentence_length, 2 * hidden_size]
        attention_weights = F.softmax(self.attention(gru_out), dim=1)  # [batch_size * max_sentence_number, max_sentence_length, 1]
        sentence_vector = torch.sum(attention_weights * gru_out, dim=1)  # [batch_size * max_sentence_number, 2 * hidden_size]
        return sentence_vector

# 定义句子级别的注意力网络
class SentenceAttention(nn.Module):
    def __init__(self, hidden_size, word_hidden_size, num_classes, dropout):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(2 * word_hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # [batch_size, max_sentence_number, 2 * hidden_size]
        attention_weights = F.softmax(self.attention(gru_out), dim=1)  # [batch_size, max_sentence_number, 1]
        document_vector = torch.sum(attention_weights * gru_out, dim=1)  # [batch_size, 2 * hidden_size]
        document_vector = self.dropout(document_vector)
        output = self.fc(document_vector)  # [batch_size, num_classes]
        return output

# 定义整体的HAN模型
class HAN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300, word_hidden_size=128, sent_hidden_size=128, max_sent_len=100, max_sent_num=1, dropout=0.1):
        super(HAN, self).__init__()
        self.word_attention = WordAttention(vocab_size, embed_dim, word_hidden_size, dropout)
        self.sentence_attention = SentenceAttention(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num

    def forward(self, x):
        # x: [batch_size, max_sentence_length]
        # 将x重塑为 [batch_size, max_sentence_number, max_sentence_length]
        x = x.view(-1, self.max_sent_num, self.max_sent_len)
        batch_size, max_sent_num, max_sent_len = x.size()
        x = x.view(batch_size * max_sent_num, max_sent_len)  # [batch_size * max_sentence_number, max_sentence_length]
        sentence_vectors = self.word_attention(x)  # [batch_size * max_sentence_number, 2 * word_hidden_size]
        sentence_vectors = sentence_vectors.view(batch_size, max_sent_num, -1)  # [batch_size, max_sentence_number, 2 * word_hidden_size]
        output = self.sentence_attention(sentence_vectors)  # [batch_size, num_classes]
        return output

def han():
    return HAN(vocab_size=95811, num_classes=4)