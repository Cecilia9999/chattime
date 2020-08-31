# chattime
QA chatbot realized by gensim and bert

## Functions

#### dataset:

- stoplistwords:  list of stop words for jieba segment

- userdict:     you can suggest some words you need to jieba

- ./data/tempo:   corpus of xiaohuangji, each line in each file is question & answers (after processing).
                  here not open these processed corpus. (files: chat0~8)



- ./data/:	faq dataset: nonghangzhidao.csv  
			

				  
#### recall: 

- 0 chatqa.py/processdata():  generate train.txt/dev.txt/test.txt for QA pairs.
							  generate simtrain.txt/simdev.txt/simtest.txt for calc similarity among questions.
							  (1 postive and 5 negative; label 1:pos, label 0:neg)
								
- 1 jiebaseg.py:  (**jieba**) cut sentences/documents as words

- 2 sentence.py:  get corpus vectors after jieba segment

- 3 similar.py:   (**Gensim**) compute similarity between input sentence and corpus.
                  offer 3 models:   **tf-idf** / **lsi** / **lda**
                
- 4 main.py:      call the above module for chat model (update: assemble chat mode into chatqa.py)

 				  
			
#### re-rank (only supports faq mode):

- 5 sim_main.py: use **pytorch-BERT** to train the model of computing similarity among questions

- 6 chatqa.py: 1) choose mode1-faq / mode2-chat 2) input your question 3) return predicted answer. 

Note:

- If chat mode, only use recall model to return the best answers. If scores < min_threshould, too much noise, re-input.

- If faq mode, use recall to get top k similar questions. 
	-  If the best scores > max_threshhould, return the best answer.
	-  If the best scores < min_threshould, too much noise, re-input.
	-  Else, use re-rank model to rank the recall item, return the best answer.  


## Use guidance
```
>>> python main.py		# only call chat mode (now replaced by chatqa.py)
>>> python chatqa.py 	# choose faq or chat mode and input question, return predicted answer.  
```
then input what you want to say, enjoy chatting time.


## Next...
improve model and fix bugs

## Reference 
chat robot: https://github.com/WenRichard/QAmodel-for-Retrievalchatbot
faq dataset: https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/

