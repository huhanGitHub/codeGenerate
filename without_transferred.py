# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataHandle import prepareData
import time
from config import EOS_token, SOS_token, Hidden_size, Teacher_forcing_ratio, MAX_LENGTH, IterTimes, r_api, r_name, dropout, Transferred_Model_Path, printTimes, Output_size, Clue_s_task_Data_path
from dataHandle import indexesFromSentence, tensorFromSentence, tensorsFromPair, asMinutes, timeSince, showPlot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=Hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size=Hidden_size, output_size=Output_size, dropout_p=dropout, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(input_tensor1, input_tensor2, input_tensor3, target_tensor, encoder1, encoder2, encoder3, decoder, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden_1 = encoder1.initHidden()
    encoder_hidden_2 = encoder2.initHidden()
    encoder_hidden_3 = encoder3.initHodden()
    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    encoder3_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor1.size(0) if input_tensor1.size(0) < input_tensor2.size(0) else input_tensor2.size(0)
    input_length = input_length if input_length < input_tensor3.size(0) else input_tensor3.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder1.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output_1, encoder_hidden_1 = encoder1(input_tensor1[ei], encoder_hidden_1)
        encoder_output_2, encoder_hidden_2 = encoder2(input_tensor2[ei], encoder_hidden_2)
        encoder_output_3, encoder_hidden_3 = encoder2(input_tensor3[ei], encoder_hidden_3)
        encoder_outputs[ei] = torch.add(encoder_output_1[0, 0], encoder_output_2[0, 0], encoder_output_3[0, 0])
        #print("encoder_outputs[ei]:",encoder_outputs[ei])

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = torch.add(encoder_hidden_1, encoder_hidden_2, encoder_output_3)
    #print("decoder_hidden.shape:", decoder_hidden.shape)

    use_teacher_forcing = True if random.random() < Teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder1_optimizer.step()
    encoder2_optimizer.step()
    encoder3_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer

def trainIters(encoder1, encoder2, encoder3, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder1_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
    encoder2_optimizer = optim.Adam(encoder2.parameters(), lr=learning_rate)
    encoder3_optimizer = optim.Adam(encoder2.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        #print(training_pair)
        input_tensor1 = training_pair[0]
        input_tensor2 = training_pair[1]
        input_tensor3 = training_pair[2]
        target_tensor = training_pair[3]

        loss, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder1_optimizer = train(input_tensor1, input_tensor2, input_tensor3, target_tensor, encoder1, encoder2,
                     encoder3, decoder, encoder1_optimizer, encoder2_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    return encoder1, encoder2, encoder3, decoder, encoder1_optimizer, encoder2_optimizer, encoder3_optimizer, decoder_optimizer


def main():
    input_lang, output_lang, pairs = prepareData(data_path=Clue_s_task_Data_path)
    print(random.choice(pairs))

    encoder1 = EncoderRNN(input_lang.n_words, Hidden_size).to(device)
    encoder2 = EncoderRNN(input_lang.n_words, Hidden_size).to(device)
    encoder3 = EncoderRNN(input_lang.n_words, Hidden_size).to(device)

    attn_decoder1 = AttnDecoderRNN(Hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, encoder2, encoder3, attn_decoder1, IterTimes, pairs, input_lang, output_lang, print_every=printTimes)

if __name__=="__main__":
    main()


