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

    file_obj = codecs.open('/content/gdrive/My Drive/nlpqa3/data/train.txt', 'r', 'utf-8' ,'ignore')
    while True:
        line = file_obj.readline()
        line = line.strip().split()
        if not line or len(line) == 0:
            break
        assert len(line) == 2, print(line)
        qList.append(line[0])
        qList_kw.append(seg.cut(line[0]))
        aList.append(line[1])
    file_obj.close()
    return qList_kw, qList, aList


def read_corpus_chat(seg):
    qList = []    
    qList_kw = []   # keyword in question
    aList = []
    
    for i in range(9):
        fn = '/content/gdrive/My Drive/nlpqa3/data/tempo/chats' + str(i)
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


# 建立倒排索引 {关键词kw：含kw的问题的idx}
def invert_table(qList_kw):
    table = {}
    for qIdx, qList in enumerate(qList_kw):
        for kw in qList:
            table[kw] = table.get(kw, []) + [qIdx]
    return table


# 在倒排表中搜索关键词，返回所有包含关键词的 QA 对
def search_invert_table(kwList, table, qList, aList):
    idxList = []
    for klist in kwList:
        for kw in klist:
            if kw in table:
                idxList.extend(table[kw])
            
    # 去掉重复的问题idx
    idxList = list(set(idxList))
    qR_List = [qList[i] for i in idxList]
    aR_List = [aList[i] for i in idxList]
    return qR_List, aR_List


def recall_topk(q, topk, mode='faq'):
    seg = Seg()
    seg.load_userdict('/content/gdrive/My Drive/nlpqa3/data/userdict')

    if mode == 'faq':
        qList_kw, qList, aList = read_corpus_faq(seg)
    elif mode == 'chat':
        qList_kw, qList, aList = read_corpus_chat(seg)
    
    # search from inversed index table
    inv_table = invert_table(qList_kw)
    qR_List, aR_List = search_invert_table(qList_kw, inv_table, qList, aList)
    
    # initialize model
    ss = SentenceSim(seg)
    ss.set_sentences(qR_List)
    ss.tfidf()        # tfidf模型
    #ss.lsi()         # lsi模型
    #ss.lda()         # lda模型
    
    top = ss.similarity_topk(q, topk)
    
    return top, qR_List, aR_List 
    

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
