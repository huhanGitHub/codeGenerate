# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataHandle import prepareData
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import EOS_token, SOS_token, Hidden_size, Teacher_forcing_ratio, MAX_LENGTH, IterTimes, r_api, r_name, dropout, Transferred_Model_Path
from dataHandle import indexesFromSentence, tensorFromSentence, tensorsFromPair,asMinutes,timeSince

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
    def __init__(self, hidden_size=128, output_size=128, dropout_p=dropout, max_length=MAX_LENGTH):
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

def indexesFromSentence(lang, sentence):
    #print(sentence)
    #print(lang.word2index)
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    pair = pair.split('\',\'')
    pair[0] = pair[0][1:]
    pair[2] = pair[2][:len(pair[2]) - 1]

    input_tensor_1 = tensorFromSentence(input_lang, pair[0])
    input_tensor_2 = tensorFromSentence(input_lang, pair[1])
    target_tensor = tensorFromSentence(output_lang, pair[2])
    return (input_tensor_1, input_tensor_2, target_tensor)

def train(input_tensor1, input_tensor2, target_tensor, encoder1, encoder2 , decoder, encoder1_optimizer, encoder2_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden_1 = encoder1.initHidden()
    encoder_hidden_2 = encoder2.initHidden()

    encoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor1.size(0) if input_tensor1.size(0) < input_tensor2.size(0) else input_tensor2.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder1.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output_1, encoder_hidden_1 = encoder1(input_tensor1[ei], encoder_hidden_1)
        encoder_output_2, encoder_hidden_2 = encoder2(input_tensor2[ei], encoder_hidden_2)
        encoder_outputs[ei] = torch.add(torch.mul(encoder_output_1[0, 0], r_api), torch.mul(encoder_output_2[0, 0], r_name))
        #print("encoder_outputs[ei]:",encoder_outputs[ei])

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = torch.add(torch.mul(encoder_hidden_1, r_api), torch.mul(encoder_hidden_2, r_name))
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
    decoder_optimizer.step()

    return loss.item() / target_length, encoder1_optimizer, encoder2_optimizer, decoder_optimizer





def trainIters(encoder1, encoder2, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder1_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
    encoder2_optimizer = optim.Adam(encoder2.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    middle=int(n_iters/2)

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        #print(training_pair)
        input_tensor1 = training_pair[0]
        input_tensor2 = training_pair[1]
        target_tensor = training_pair[2]

        loss, encoder1_optimizer, encoder2_optimizer, decoder1_optimizer = train(input_tensor1, input_tensor2, target_tensor, encoder1, encoder2,
                     decoder, encoder1_optimizer, encoder2_optimizer, decoder_optimizer, criterion)
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
    return encoder1, encoder2, decoder, encoder1_optimizer, encoder2_optimizer, decoder_optimizer

def evaluate(encoder1, encoder2, decoder, sentence1, sentences2, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor1 = tensorFromSentence(input_lang, sentence1)
        input_tensor2 = tensorFromSentence(input_lang, sentences2)

        input_length = input_tensor1.size(0) if input_tensor1.size(0) < input_tensor2.size(0) else input_tensor2.size(0)

        encoder_hidden_1 = encoder1.initHidden()
        encoder_hidden_2 = encoder2.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder1.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output_1, encoder1_hidden = encoder1(input_tensor1[ei], encoder1_hidden)
            encoder_output_2, encoder2_hidden = encoder2(input_tensor2[ei], encoder2_hidden)
            encoder_outputs[ei] += torch.add(torch.mul(encoder_output_1[0, 0], r_api),
                                            torch.mul(encoder_output_2[0, 0], r_name))

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = torch.add(torch.mul(encoder_hidden_1, r_api), torch.mul(encoder_hidden_2, r_name))

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder1, encoder2, decoder, pairs, input_lang, output_lang, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder1, encoder2, decoder, pair[0],input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate2AndShowAttention(encoder1, encoder2, attn_decoder1, input_sentence, input_lang, output_lang):
    output_words, attentions = evaluate(
        encoder1, encoder2, attn_decoder1, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    #showAttention(input_sentence, output_words, attentions)

######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#
plt.switch_backend('agg')
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def main():
    input_lang, output_lang, pairs = prepareData()
    print(random.choice(pairs))

    encoder1 = EncoderRNN(input_lang.n_words, Hidden_size).to(device)
    encoder2 = EncoderRNN(input_lang.n_words, Hidden_size).to(device)

    attn_decoder1 = AttnDecoderRNN(Hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    encoder1, encoder2, attn_decoder1, encoder1_optimizer, encoder2_optimizer, decoder1_optimizer = trainIters(
        encoder1, encoder2, attn_decoder1, IterTimes, pairs, input_lang, output_lang, print_every=5000)

    torch.save({
        'Encoder1_state_dict': encoder1.state_dict(),
        'Encoder2_state_dict': encoder2.state_dict(),
        'Decoder_state_dict': attn_decoder1.state_dict(),
        'encoder1_optimizer': encoder1_optimizer,
        'encoder2_optimizer': encoder2_optimizer,
        'decoder1_optimizer': decoder1_optimizer,
        'input_lang': input_lang,
        'output_lang': output_lang,
        }, Transferred_Model_Path)

if __name__=="__main__":
    main()


