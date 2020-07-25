# -*- coding: utf-8 -*-
from jiebaseg import Seg


class Sentence:

    def __init__(self, sentence, seg, id=0):
        self.id = id
        self.origin_sentence = sentence
        self.cuted_sentence = self.cut(seg)

    # jieba cut word
    # seg = Seg()
    def cut(self, seg):
        return seg.cut_for_search(self.origin_sentence)

    # get cuted setence list
    def get_cuted_sentence(self):
        return self.cuted_sentence

    # get original sentence
    def get_origin_sentence(self):
        return self.origin_sentence

    # set socre of this sentence, here socre=similarity
    def set_score(self, score):
        self.score = score
