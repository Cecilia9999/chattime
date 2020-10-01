# -*- coding: utf-8 -*-

import sys
import os
import gc
import random
import pandas as pd
import re
import numpy as np

from nltk.probability import FreqDist
import time
import jieba
import codecs

from jiebaseg import *
from similar import SentenceSim
import sim_main
import recall_main

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

#import pymysql
from tqdm import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_data():
    data_dir = '/content/gdrive/My Drive/nlpqa3/data'
    file_name = 'nonghangzhidao.csv'
    file_name2 = 'nonghangzhidao3.txt'
    file_path_name = os.path.join(data_dir, file_name)
    file_path_name2 = os.path.join(data_dir, file_name2)
    out = '/content/gdrive/My Drive/nlpqa3/data/sim.txt'
    out2 = '/content/gdrive/My Drive/nlpqa3/data/train.txt'
    out3 = '/content/gdrive/My Drive/nlpqa3/data/test.txt'
    out4 = '/content/gdrive/My Drive/nlpqa3/data/dev.txt'

    q_list, a_list = [], []
    maxlen = 0
    with open(file_path_name2, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split('\t')
            line0 = re.sub(r"(\xe2\x98\x85|\xe2\x97\x86)|[_$^*~#￥%……&*]", '', line[0])
            line1 = re.sub(r"(\xe2\x98\x85|\xe2\x97\x86)|[_$^*~#￥%……&*]", '', line[1])
            q_list.append(line0)
            a_list.append(line1)
            maxlen = max(maxlen, len(line0))
    f.close()
    print(maxlen)

    train_size = int(len(q_list) * 0.7)
    val_size = train_size + int((len(q_list) - train_size) * 0.5) 
    print(train_size, val_size, len(q_list)-val_size)

    with open(out2, "w", encoding='utf-8') as f1:
        for i in range(train_size):
            f1.write(q_list[i] + ' ' + a_list[i] + '\n')
    f1.close()

    with open(out4, "w", encoding='utf-8') as f4:
        for i in range(train_size, val_size):
            f4.write(q_list[i] + ' ' + a_list[i] + '\n')
    f4.close()

    with open(out3, "w", encoding='utf-8') as f2:
        for i in range(val_size, len(q_list)):
            f2.write(q_list[i] + ' ' + a_list[i] + '\n')
    f2.close()

    for s,e,out in zip([0, train_size, val_size], [train_size, val_size, len(q_list)], ['simtrain.txt', 'simdev.txt','simtest.txt']):
        q_lista = q_list[s:e]

        res = []
        p_n = {}
        for q in q_lista:
            pos, neg = [], []
            pos.append(q)
            while True:
                neg = random.sample(q_lista, 1)
                if pos[0] not in neg:
                    break
            res.append(pos+neg)

        cnt = 0

        with open(os.path.join(data_dir, out), "w", encoding='utf-8') as f3:
            for q in res:
                assert len(q) == 2
                for j in range(len(q)):
                    if j == 0:
                        f3.write(str(cnt) + '\t' + str(q[0]) + '\t' + str(q[0]) + '\t' + '1' + '\n')
                    else:
                        f3.write(str(cnt) + '\t' + str(q[0]) + '\t' + str(q[j]) + '\t' + '0' + '\n')
                    cnt += 1
        f3.close()


#file_path = '/content/gdrive/My Drive/nlpqa3/data/'
#file = ['train.txt', 'test.txt']

# model_name=os.path.join(args_test["load_path"], 'pytorch_model.bin')
def load_sim_model(config_file, model_name, label_num=2):
    bert_config = BertConfig.from_pretrained(config_file)
    bert_config.num_labels = label_num
    model_kwargs = {'config': bert_config, "from_tf": False}
    model = BertForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    return model


def semantic_match(model, tokenizer, question, attribute_list, max_seq_length):
    batch_size = 128
    all_tokens = []
    all_masks = []
    all_segments = []
    features = []
    for attribute in attribute_list:

        textA = tokenizer.tokenize(question)
        textB = tokenizer.tokenize(attribute)
        idsA = tokenizer.convert_tokens_to_ids(textA)
        idsB = tokenizer.convert_tokens_to_ids(textB)

        # cls + idsA + sep + idsB + sep
        input_ids = tokenizer.build_inputs_with_special_tokens(idsA, idsB)
        masks = [1] * len(input_ids)
        #token_type_ids = tokenizer.create_token_type_ids_from_sequences(idsA, idsB)
        token_type_ids = [0] * (len(idsA) + 2) + [1] * (len(idsB) + 1)
        
        # pad seq to max length
        pad_seq = [0] * (max_seq_length - len(input_ids))
        input_ids += pad_seq
        masks += pad_seq
        token_type_ids += pad_seq

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(masks) == max_seq_length, "Error with input length {} vs {}".format(len(masks), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids), max_seq_length)

        features.append(
            sim_main.SimInputFeatures(input_ids, masks, token_type_ids)
        )

       
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    assert all_input_ids.shape == all_attention_mask.shape
    assert all_attention_mask.shape == all_token_type_ids.shape


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    data_num = all_attention_mask.shape[0]
    
    all_logits = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': None
                     }

            outputs = model(**inputs)
            
            logits = outputs[0]

            #logits = logits.sigmoid(dim = -1)
            #logits = logits.softmax(dim = -1)

            if all_logits is None:
                all_logits = logits.clone()
            else:
                all_logits = torch.cat([all_logits, logits], dim = 0)

    prediction = all_logits.argmax(dim = -1)
    if prediction.sum() == 0:
        return torch.tensor(-1)
    else:
        return prediction.argmax(dim = -1)


if __name__ == "__main__":
    # get train, dev, test data, save as '.txt'  
    
    max_eps = 0.9
    min_eps = 0.2
    sim_path = '/content/gdrive/My Drive/nlpqa3/output/data/'
    file_path = '/content/gdrive/My Drive/nlpqa3/data/'
    
    #seg = Seg()
    #seg.load_userdict(os.path.join(file_path, 'userdict'))

    while True:
        mode = input("金融知识faq扣1；闲聊扣2: (退出请输入：q)")
        if mode == 'q':
            break

        if mode == '1':
            m = 'faq'
            q = input("请输入您的问题: (退出请输入：q)")
        elif mode == '2':
            m = 'chat'
            q = input("请输入您的问题: (退出请输入：q)")
        else:
            input("请输入正确选项！")
            continue

        if q == 'q':
            break
        
        time1 = time.time()
        
        # q_topk: ([k_questions_index,...], [k_questions_score, ...])
        # q_topk[0][0]: get first index in all index of topk questions
        q_topk, qList, aList = recall_main.recall_topk(q, 5, mode=m)

        # td-idf
        if q_topk[1][0] > max_eps: 
            print("我的回答: {} (分数:{})".format(aList[q_topk[0][0]], q_topk[1][0]))
            for idx, score in zip(*q_topk):
                if idx == 0: continue
                print('其他相关问题:\n{},  socre: {}'.format(qList[idx], score))
        
        # too much noise, re-input
        elif q_topk[1][0] < min_eps: 
            print("抱歉！我不太理解您的意思...")
            continue
        
        # re-rank model (only for faq mode)
        else:
            if m == 'chat':  # not use re-rank  
                print("我的回答: {} (分数:{})".format(aList[q_topk[0][0]], q_topk[1][0]))
                for idx, score in zip(*q_topk):
                    if idx == 0: continue
                    print('其他相关问题:\n{},  socre: {}'.format(qList[idx], score))   
                continue

            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', return_tensors='pt')
            sim_processor = sim_main.preProcessor()
            sim_model = load_sim_model(config_file=os.path.join(sim_path, 'config.json'),
                                        model_name=os.path.join(sim_path, 'pytorch_model.bin'),
                                        label_num=len(sim_processor.get_labels()))

            sim_model = sim_model.to(device)
            sim_model.eval()

            ql = [qList[i] for i in q_topk[0]]
            a_idx = semantic_match(sim_model, tokenizer, q, ql, 64).item()
            if a_idx == -1:
                res = ''
            else:
                print("My answers: {}". format(aList[q_topk[0][a_idx]]))        
        
        time2 = time.time()
        print('Time: ', time2-time1)