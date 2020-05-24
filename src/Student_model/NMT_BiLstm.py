import torch
import random
import time
import math
import torch.nn as nn
import Pre_train_data
import show
import Prepare_Data
import Attention_decoder
import EncoderRNN

teacher_forcing_ratio = 0.5
MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

class Bil_lstm(object):
    def __init__(self, data, params):



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # pay attention to the size(0), it maybe length of sentences, or it maybe batch_size
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # get the length of input and output
    # print("input_tensor", input_tensor)
    # print("target_tensor", target_tensor)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)
    # init the output of encoder, must be max_length, because of the output is the input of decoder

    loss = 0
    for ei in range(input_length):

        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # pay attention to the input of encoder, it must be [length, batch_size, n_words]
        # n_words means the ont hot output
        # print("encoder_output", encoder_output)
        encoder_outputs[ei] = encoder_output[0, 0]
        # print(encoder_outputs[ei])
        # encoder_outputs 变成了三维是为什么呢

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = decoder.initHidden()
    # 初始化第一个 decoder的输入

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    # init the optimizer
    # training_pairs = [Pre_train_data.tensorFromPair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = [Pre_train_data.tensorFromPair(input_lang, output_lang,random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()
    # 交叉熵损失函数
    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # input_tensor 是 例如[3, 4, 5, 0] 的 tensor
        # print(input_tensor.size())
        # print(target_tensor.size())
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


hidden_size = 1024

encoder1 = EncoderRNN.EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = Attention_decoder.Attention_decoder(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1 , 75000, print_every=5000)

# encoder = EncoderRNN.EncoderRNN(128, 128).cuda()
# decoder = Attention_decoder.Attention_decoder(128, 128).cuda()
#
# trainIters(encoder, decoder, n_iters=100)
