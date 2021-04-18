import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from utils import *
from nltk.stem.porter import PorterStemmer
import nltk
import math
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords



nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
stoplist = stopwords.words('english')
porter_stemmer = PorterStemmer()
stoplist = ['the','a','no','if','an','and','but','is','are','be','were','in','wchich','of','for','.','!',',','?','that','not','this']
model = Doc2Vec.load('doc2vec.bin')




def deal(p,t):
    
    p = p.split()
    if p[0] in pre_word:
        p=p[1:]
    b = ' '.join(p)
    tag = nltk.pos_tag(p)
    if len(p)<=2:
        return set([b])
    else:
        pro=2
    
    if p[0] in stoplist:
        p = p[1:]
    if len(p)==1:
        return set([b])

    ret=[b]
    for e in p:
        if e not in idf:
            return set(ret)
    
    if tag[0][1] not in ['NN','NNS','NNP']: 
        r0=idf[p[0]] * t.count(p[0])
        r1=idf[p[1]] * t.count(p[1])
        if r0*5 < r1:
            ret=[]
        ret.append(' '.join(p[1:]))
        return set(ret)

    if idf[p[-1]]*t.count(p[-1])*5 < idf[p[-2]]*t.count(p[-2]):
        ret.append(' '.join(p[:-1]))
        return set(ret)
    
    return set(ret)




def Extract(input):

    can_list,can_set = get_ngram(input)
    idf = np.load('word_dic.npy', allow_pickle=True).item()

    new_bb=set()
    if False:
        for i in range(len(all_can)):
            t=all_can[i]
            tmp2=set()
            for p in t:
                tmp2 = tmp2 | deal(p,a[i])

            all_can[i] = tmp2
            new_bb = new_bb | tmp2

    record = []
    idf_p = np.load('phrase_dic.npy', allow_pickle=True).item()
    
    for i,e in enumerate(input):

        pre = e.lower()
      
        pre_list = nltk.tokenize.word_tokenize(pre) 
        stem_pre = [porter_stemmer.stem(q) for q in pre_list]
        stem_pre = ' '.join(stem_pre)

   
        doc_emb = model.infer_vector(pre_list)
        doc_emb = doc_emb / math.sqrt(sum([doc_emb[k]*doc_emb[k] for k in range(300)])) 
        rank=[]
        rank2 = []
        l = len(pre.split('.'))
        absent_can = set()
        for phrase in can_set:
            phrase = phrase.split()
            flg = 0
            for w in phrase:
                if w not in pre+list:
                    flg = 1
                    break
            if flg==0:
                absent_can.add(phrase)
                    
                
        for j,q in enumerate(list(can_list[i])+list(absent_can)):
 
            if q not in idf:
                continue

            q_list = nltk.tokenize.word_tokenize(q) 
            emb = model.infer_vector(q_list)
            emb = emb / math.sqrt(sum([emb[k]*emb[k] for k in range(300)])) 
          
            emb = emb.reshape([1,300])

            sim = float(np.dot(doc_emb.reshape([1,300]), emb.reshape([300,1])))
            if l>10:
                sim =  pre.count(q)*idf[q] * sim 
            if j < len(can_list[i]):
                rank.append([sim,q])
            else:
                ran2k.append([sim2,q])


   
        rank.sort(reverse=True)
        rank2.sort(reverse=True)

        rank = reduce(rank)
        rank2 = reduce(rank2)

        record.append([input[i], list(set(rank[:5] + rank2[:5])))
    np.save('silver.npy', record)
 
if __name__ == '__main__':
    Extract(list(np.load('document.npy', allow_pickle=True)))




