import gc
import tqdm
import numpy as np
from gensim import corpora, models, similarities
from sentence import Sentence
from collections import defaultdict
from gensim.test.utils import datapath, get_tmpfile
import os

class SentenceSim:
    def __init__(self, seg):
        self.seg = seg
    
    def set_sentences(self, sentences):
        self.sentences = []
        for i in range(0, len(sentences)):
            self.sentences.append(Sentence(sentences[i], self.seg, i))
    
    def get_cuted_sentences(self):
        cuted_sentences = []
        for sen in self.sentences:
            cuted_sentences.append(sen.get_cuted_sentence())
        
        #print('s: ', cuted_sentences)
        return cuted_sentences

    def simple_model(self, min_freq=1):
        self.texts = self.get_cuted_sentences()
        
        # count freq of words
        freq = defaultdict(int)
        
        for text in self.texts:
            for token in text:
                freq[token] += 1
        
        # delete low_freq words
        self.texts = [[i for i in text if freq[i] > min_freq] for text in self.texts]

        # create dict of list of words
        self.dicts = corpora.Dictionary(self.texts)
        # convert words to bow   bow: [(word_id, word_freq)], sparse vec
        # order: word(after jieba) -> bow(= word freq) -> vec(diff model)
        self.corpus_simple = [self.dicts.doc2bow(text) for text in self.texts]
        #print('simple: ', self.corpus_simple)


    # tf-idf model
    def tfidf(self):
        # initial 
        self.simple_model()

        # fit model
        self.model = models.TfidfModel(self.corpus_simple)

        # apply model to corpus
        self.corpus = self.model[self.corpus_simple]
        # [[(0, 1), (1, 3), (2, 1), (3, 1), (4, 1)], [(0, 1), (1, 2), (3, 1), (4, 1), (5, 1)], [(0, 1), (5, 1), (6, 1)]]
        # [[sent1 (word0_id, word0_freq),()], [sent2 (), (), ()], [sent3 (), ()] ]

        # create similarity matrix
        # self.index = similarities.MatrixSimilarity(self.corpus)

        if os.path.exists('./chat.index'):
            self.index = similarities.Similarity.load('./chat.index')
        else:
            output_fname = get_tmpfile("saved_index")
            self.index = similarities.Similarity(output_fname, self.corpus, len(self.dicts))
            self.index.save('./chat.index')
        

    # lsi model 
    def lsi(self):
        self.simple_model()

        # fit model
        self.model = models.LsiModel(self.corpus_simple, num_topics=200)
        self.corpus = self.model[self.corpus_simple]
        # self.corpus: [(w0_id, w0_tdf), (w1_id, w1_tdf), ... ]
        
        # create similarity matrix
        # self.index = similarities.MatrixSimilarity(self.corpus)
    
        if os.path.exists('./chat2.index'):
            self.index = similarities.Similarity.load('./chat2.index')
        else:
            output_fname = get_tmpfile("saved_index2")
            #print('p1')
            self.index = similarities.Similarity(output_fname, self.corpus, 200)
            #print('p2')
            self.index.save('./chat2.index')
        

    # lda model 
    def lda(self):
        self.simple_model()

        # fit model
        self.model = models.LdaModel(self.corpus_simple, num_topics=50)
        self.corpus = self.model[self.corpus_simple]

        # create similarity matrix
        #self.index = similarities.MatrixSimilarity(self.corpus)
        if os.path.exists('./chat3.index'):
            self.index = similarities.Similarity.load('./chat3.index')
        else:
            output_fname = get_tmpfile("saved_index3")
            print('p1')
            self.index = similarities.Similarity(output_fname, self.corpus, 50)
            print('p2')
            self.index.save('./chat3.index')


    # handle new input sentence
    def sen2vec(self, sentence):
        sentence = Sentence(sentence, self.seg)
        # word in current sentence -> bow (= word freq)
        vec_bow = self.dicts.doc2bow(sentence.get_cuted_sentence())
        return self.model[vec_bow]  # apply model to bow, convert bow to vec
    
    # create matrix vec to include all []
    def bow2vec(self):
        vec = []
        len = max(self.dicts) + 1       # get max length of sentence in doc
        for content in self.corpus:     # sentence in doc
            sentence_vec = np.zeros(len)
            for co in content:          # word in doc   
                # content[(co[0], co[1])]     
                # co[0] = word_id, co[1] = word_tdf
                sentence_vec[co[0]] = co[1]
            vec.append(sentence_vec) 
            #print('bow2vec: ', vec)
        return vec

    # get similarity between input sentence and corpus
    def similarity(self, sentence):
        sentence_vec = self.sen2vec(sentence)

        sims = self.index[sentence_vec]
        # [0.70837465(sim of input & sent1 in corpus), 0.124237(sim of input & sent2), ... ]

        # get sent in corpus with max sim 
        sim = max(enumerate(sims), key=lambda x: x[1])
        
        index = sim[0]
        score = sim[1]
        sentence = self.sentences[index]   # get this sent's object in corpus
                                           # self.sentences=[list of Sentence object] 
        sentence.set_score(score)          # set this sent's object's score 
        return sentence                    # return an object

    # get former k similarity between input sentence and corpus
    def similarity_topk(self, sentence, k):
        sentence_vec = self.sen2vec(sentence)

        sims = self.index[sentence_vec]
        # [0.70837465(sim of input & sent1 in corpus), 0.124237(sim of input & sent2), ... ]

        # get sent in corpus with max sim
        #sim = max(enumerate(sims), key=lambda x: x[1])

        sim_k = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
        
        indices = [sim[0] for sim in sim_k]
        scores = [sim[1] for sim in sim_k]
        '''
        sentences = []
        for index, socre in zip(indices, socres)
            objs = self.sentences[index]
            objs.set_score(score)
            sentences.extend(objs)   
        return sentences
        '''
        return indices, scores  # return indices cuz it's convenient for search corresponding answers

 
