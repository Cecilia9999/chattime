# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time

from jiebaseg import *
from similar import SentenceSim

def read_corpus_faq(seg):
    qList = []    
    qList_kw = []   # keyword in question
    aList = []

    data = pd.read_csv('./data/qa_.csv', header=None)
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(t[0])
        qList_kw.append(seg.cut(t[0]))
        aList.append(t[1])
    return qList_kw, qList, aList

def read_corpus_chat(seg):
    qList = []    
    qList_kw = []   # keyword in question
    aList = []
    
    for i in range(9):
        fn = './tempo/chats' + str(i)
        with open(fn, 'r', encoding='utf-8') as f2:
            for lines in f2:
                if lines.strip() == '': break 
                line = lines.split()        # format [q, a]
                if len(line) < 2:
                    continue
                qList.append(line[0])
                qList_kw.append(seg.cut(line[0]))
                aList.append(line[1])
     
    return qList_kw, qList, aList


if __name__ == "__main__":
    seg = Seg()
    seg.load_userdict('./userdict')

    # read data
    qList_kw, qList, aList = read_corpus_chat(seg)
    
    # initialize model
    ss = SentenceSim(seg)
    ss.set_sentences(qList)
    # ss.tfidf()         # tfidf模型
    # ss.lsi()         # lsi模型
    ss.lda()         # lda模型

    while True:
        q = input("Your question, please: (exit: 'q')")
        if q == 'q':
            break
        time1 = time.time()

        # q_topk: ([k_questions_index,...], [k_questions_score, ...])
        # q_topk[0][0]: get first index in all index of topk questions
        q_topk = ss.similarity_topk(q, 5)
        print("My answers: {}". format(aList[q_topk[0][0]]))

        for idx, score in zip(*q_topk):
            print('same questions: {},  socre: {}'.format(qList[idx], score))
        
        time2 = time.time()
        print('Time: ', time2-time1)
