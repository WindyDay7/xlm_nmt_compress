import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10


class Attention_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length = MAX_LENGTH):
        super(Attention_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True,batch_first=True)
        self.out_1 = nn.Linear(2*self.hidden_size, self.output_size)
        self.out = nn.Linear(self.output_size, self.output_size)
        # 这里是定义了这些参数矩阵, 模型本质的未知数是矩阵的参数, 所以本质是定义矩阵
        # 因此在下面的调用中, 没有使用像 tensorflow 一样的显式的相乘, 而是直接调用

    def forward(self, in_put, hidden, encoder_outputs):
        embedded = self.embedding(in_put).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # dim=1 表示安行计算
        # 这里是 现将 embedded 和 hidden 两个横向拼接起来, 变成了二维的tensor 的矩阵, cat 里面的1 是最深层的意思
        # 然后再调用 nn.Linear 计算,
        # attn_weights的结果是 d_model * max_length 维度
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsequence(0))
        # unsequence 就是对tensor 添加维度, 因为我猜想 softmax 的计算结果加一维变成3维矩阵,
        # 然后就是 torch.mm 其实就是算了下attention 的结果, 结果是三维的,
        # encoder_outputs 是 max_length * hiden_size 这个很好理解, 因此 attn_applied 还是 d_model * hiden_size 维度
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # 再结合 encoder_outputs 做一次 attention
        output = F.relu(output)
        # 常用的激活函数
        output, hidden = self.lstm(output, hidden)
        # output = 2 * hidden
        output = self.out_1(output)
        output = F.log_softmax(self.out(output[0]), dim=1)
        # 这也是安行拼接, 也就是最后的得分了
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)