# -*- coding:utf-8 -*-
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataHandle import filterPair
import  re

def filterString(s):
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    #s = re.sub(r"[^a-zA-Z.!?,'\._!=]\+-\*/%\|&", r" ", s)
    s=s.replace(" +"," ")
    return s

def main():
    #bleu('save/compare.txt')

    ref = "let it go".split()
    hyp = "let go it".split()
    cc = SmoothingFunction()
    score = sentence_bleu([ref], hyp, smoothing_function=cc.method4)

    print("The bleu score is: " + str(score))


if __name__ == '__main__':
    main()


