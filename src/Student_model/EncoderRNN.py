import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        # with embedding, the input size of LSTM is hidden_size
        self.lstm = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.out = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, in_put, hidden, Batch_size):
        embedded = self.embedding(in_put).view(1, Batch_size, -1)
        output = embedded
        # 中间的 embed 向量就是通过编码后的 one-hot向量
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_size).cuda()
    # 这里为什么要设置成三维的呢

