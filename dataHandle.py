# -*- coding: utf-8 -*-
import torch
import unicodedata
import re
import pickle
import random
from config import Data_path, Test_ratio, MAX_LENGTH

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

######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

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
    return s

def readData(path):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path, encoding='utf-8').\
        read().strip().split('\n')

    for i in range(len(lines)):
        lines[i]=normalizeString(lines[i])
        lines[i]=lines[i][1:len(lines[i])-1]

    input_lang = Dictionary('input')
    output_lang = Dictionary('output')

    return input_lang, output_lang, lines

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    #print(p)
    pp=p.split('\',\'')
    return len(pp[0].split(' ')) < MAX_LENGTH and \
        len(pp[1].split(' ')) < MAX_LENGTH and len(pp[2].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData():
    input_lang, output_lang, pairs = readData(Data_path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for line in pairs:
        pair=line.split('\',\'')
        pair[0]=pair[0][1:]
        pair[2]=pair[2][:len(pair[2])-1]
        input_lang.addSentence(pair[0].strip())
        input_lang.addSentence(pair[1].strip())
        #print(pair)
        output_lang.addSentence(pair[2].strip())

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    with open('save/pairs.txt','w', encoding='utf-8') as f:
        f.write(str(pairs))

    with open('save/input.dic','wb') as f:
        pickle.dump(input_lang, f)
        #f.write(input_str)

    with open('save/output.dic','wb') as f:
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

def main():
    prepareData()

if __name__=='__main__':
    main()