import torch
import random
import torch.utils.data as data
import os
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence
import pickle
import random
import nltk


class MyDataset(data.Dataset):
  
    def __init__(self, data_name='silver.npy', vocab_name='vocab_kp20k.npy',cls2=0):
        
        self.f = list(np.load(data_name, allow_pickle=True))
        
        self.vocab = np.load(vocab_name, allow_pickle=True).item()
    
        
         

    def __getitem__(self, index):

        
        x, trg= self.f[index]
        x = x.lower()
        x = nltk.tokenize.word_tokenize(x)
        
    
        for i in range(len(x)):
            x[i] = self.vocab(x[i])
        x.append(self.vocab('<end>'))
        x = [self.vocab('<start>')] + x

        if len(x)>512:
            x = x[:512]
        while len(x) < 512:
            x.append(self.vocab('<pad>'))

        src = torch.Tensor(x)
 

        x = trg
        x = ','.join(x)
        x = nltk.tokenize.word_tokenize(x.lower())
        
    
        for i in range(len(x)):
            x[i] = self.vocab(x[i])
        x.append(self.vocab('<end>'))
        x = [self.vocab('<start>')] + x

        while len(x) < 30:
            x.append(self.vocab('<pad>'))

        trg = torch.Tensor(x)
        
 
        return src, trg

    def __len__(self):
        return  len(self.f)

