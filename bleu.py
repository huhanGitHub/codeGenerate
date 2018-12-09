# -*- coding:utf-8 -*-
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re


def filterString(s):
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    #s = re.sub(r"[^a-zA-Z.!?,'\._!=]\+-\*/%\|&", r" ", s)
    s=s.replace(" +"," ")
    return s

def bleu(path):

    cc=SmoothingFunction()
    with open(path, 'r') as f:
        line=f.readlines()
        score=0
        count=0
        for pair in line:
            #print(line)
            pair_s = pair.split('|||||')
            if len(pair_s) < 2:
                break
            reference = filterString(pair_s[0])
            candidate = filterString(pair_s[1])
            if reference == ' ' or candidate == ' ':
                continue
            print(reference+"-----"+candidate)
            score += sentence_bleu([reference.strip().split(' ')], candidate.strip().split(' '), smoothing_function=cc.method3)
            #score += sentence_bleu([reference.strip().split(' ')], candidate.strip().split(' '), weights=[1, 0, 0, 0])
            count+=1

        score/=count
        print("The bleu score is: "+str(score))

def main():
    bleu('save/compare.txt')

if __name__ == '__main__':
    main()


