# -*- coding: utf-8 -*-
import torch
import unicodedata
import re
import pickle
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import Clue_s_task_Data_path, Test_ratio, MAX_LENGTH, EOS_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?,'\._!=]\+-\*/%\|&", r" ", s)
    #s = re.sub(r'[^a-zA-Z0-9. ]+', r'', s)
    return s

def readData(path):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path, encoding='utf-8').\
        read().strip().split('\n')

    for i in range(len(lines)):
        lines[i]=normalizeString(lines[i])
        lines[i]=lines[i][1:len(lines[i])-1]

    input_lang = Dictionary('Input Dic')
    output_lang = Dictionary('Output Dic')

    return input_lang, output_lang, lines

def filterPair(p):
    #print(p)
    pp=p.split('\',\'')
    return len(pp[1].split(' ')) < MAX_LENGTH and len(pp[2].split(' ')) < MAX_LENGTH and len(pp[3].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(data_path=Clue_s_task_Data_path):
    input_lang, output_lang, pairs = readData(data_path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)

    print("Counting words...")
    for i in range(len(pairs)):
        pair=pairs[i].split('\',\'')
        pair[0] = re.sub(r'[^a-zA-Z0-9. ]+', r'', pair[0])
        pair[1] = re.sub(r'[^a-zA-Z0-9. ]+', r'', pair[1])
        pair[2] = re.sub(r'[^a-zA-Z0-9. ]+', r'', pair[2])
        pair[3] = re.sub(r'[^a-zA-Z0-9. ]+', r'', pair[3])

        #pair[0]=pair[0][1:]
        #pair[2]=pair[2][:len(pair[2])-1]
        input_lang.addSentence(pair[0].strip())
        input_lang.addSentence(pair[1].strip())
        input_lang.addSentence(pair[2].strip())
        #print(pair)
        output_lang.addSentence(pair[3].strip())
        s=pair[0]+','+pair[1]+','+pair[2]+','+pair[3]
        pairs[i]=s

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    with open('save/pairs.txt', 'w', encoding='utf-8') as f:
        f.write(str(pairs))

    with open('save/input.dic', 'wb') as f:
        pickle.dump(input_lang, f)
        #f.write(input_str)

    with open('save/output.dic', 'wb') as f:
        pickle.dump(output_lang, f)
        #f.write(output_str)

    with open('save/input.dic','rb') as f:
      input=pickle.load(f)

    with open('save/output.dic','rb') as f:
      output=pickle.load(f)

    random.shuffle(pairs)
    train = pairs[:len(pairs)-(int)(len(pairs)*Test_ratio)]
    test = pairs[len(pairs)-int(len(pairs)*Test_ratio):]

    with open('save/train_pairs.txt', 'w', encoding='utf-8') as f1, open('save/test_pairs.txt', 'w', encoding='utf-8') as f2:
        for i in range(len(train)):
            f1.write(str(train[i]))
            f1.write('\n')

        for i in range(len(test)):
            f2.write(str(test[i]))
            f2.write('\n')

    return input_lang, output_lang, train

def indexesFromSentence(lang, sentence):
    # print(sentence)
    # print(lang.word2index)
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    #pair=pair.replace('\\\\', '')
    pp = pair.split(',')

    # print(pp[0])
    # print(pp[1])
    # print(pp[2])
    # print(pp[3])

    input_tensor_1 = tensorFromSentence(input_lang, pp[0])
    input_tensor_2 = tensorFromSentence(input_lang, pp[1])
    input_tensor_3 = tensorFromSentence(input_lang, pp[2])
    target_tensor = tensorFromSentence(output_lang, pp[3])
    return (input_tensor_1, input_tensor_2, input_tensor_3, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

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
    prepareData()

if __name__=='__main__':
    main()