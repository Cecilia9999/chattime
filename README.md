# chattime
chatting robot realized by gensim

## Functions
- jiebaseg.py:  use jieba to cut sentences/documents as words

- sentence.py:  get corpus vectors after jieba segment

- similar.py:   compute similarity between input sentence and corpus.
                offer 3 models:   tf-idf / lsi / lda
                
- main.py:      call the above module to deal with 

- tempo:        corpus of xiaohuangji, each line in each file is question & answers (after processing).
                here not open these processed corpus.
                
- stoplistwords:  list of stop words for jieba segment

- userdict:     you can suggest some words you need to jieba


## Use guidance
```
>>> python main.py
```
then input what you want to say, enjoy chatting time.


## Next...
construct chat robt via deep learning(NLP)

## Reference 
chat robot: https://github.com/WenRichard/QAmodel-for-Retrievalchatbot

