import torch.nn as nn
import nltk
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import random
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim=42153, emb_dim=200, hid_dim=256, dropout=0.5,name='emb_kp20k.npy'):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!
        
        #emb = np.load(name)
        #self.embedding.weight.data.copy_(torch.from_numpy(emb))
    
        
        self.rnn = nn.LSTM(emb_dim, hid_dim,bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded) #no cell state!

        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
    
        return hidden
    



class Decoder(nn.Module):
    def __init__(self, output_dim=24487, emb_dim=200, hid_dim=256, dropout=0.5, name='emb_kp20k2.npy'):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

     
        #emb = np.load(name)
        #self.embedding.weight.data.copy_(torch.from_numpy(emb))
    

        
        self.rnn = nn.LSTM(emb_dim, hid_dim,bidirectional=True)
        
        self.fc_out = nn.Linear(emb_dim + hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        #emb_con = torch.cat((embedded, context), dim = 2)
            
        #emb_con = [1, batch size, emb dim + hid dim]
            
        output, hidden = self.rnn(embedded, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        output = torch.cat((embedded.squeeze(0), (0.5*hidden[0][0][:].squeeze(0)+hidden[0][1][:])), 
                           dim = 1)
        
        #output = [batch size, emb dim + hid dim * 2]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.beam = 30
        self.vocab = np.load('vocab_kp20k.npy', allow_pickle=True).item()
        self.link = np.load('link.npy', allow_pickle=True).item()

    def forward(self, src, trg, info, teacher_forcing_ratio = 0.5):
    
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        if teacher_forcing_ratio==0:
            trg_len = 6
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is the context
        context = self.encoder(src)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        back = hidden
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        m = nn.Softmax(dim=1)
    
        ret = []
        first_out, first_hidden = self.decoder(input, hidden, context)
        first_out[0,3] = -1
        first_out[0,0] = -1
        
        first_out = m(first_out)
        #print(info)
        n=set([1])
        for e in info:
            if int(e)!=3:
                n.add(int(e))
        info = n
        cnt=0
        for e in info:
            first_out[0,e] = float(first_out[0,e])*3
        
        for i in range(15):          
            input = first_out.argmax(1)
           
            if first_out[0, input] == -1:
                break
            
            first_out2, first_hidden2 = self.decoder(input, first_hidden, context)
            first_out2 = m(first_out2)
            first_out2[0,3] = -1
            first_out2[0,input] = -1
            word = self.vocab.idx2word[int(input)]
            ret.append([float(first_out[0,input]), first_out2, first_hidden2, int(input), [word]]) 
            first_out[0,input]= -1
        cc=0
        for j in range(5):
            tmp = []
            for e in ret: 
                if (j == 0) and (cc<=2):
                    e[1][0,1] = -1
                    cc += 1
                
                if e[-1][-1]=='<end>':
                    tmp.append(e)
                    continue
            
                t = info & self.link[e[-2]]
                for p in t:
                    e[1][0,p] = e[1][0,p]*5
                #or p in info:
                #    e[1][0,p] *= 2
                for i in range(2): 
                    input = e[1].argmax(1)
                
                    """
                    while int(input) not in t:
                        e[1][0, input] = -1
                        input = e[1].argmax(1) 
                        if e[1][0,input] == -1:
                            break
                    """
                    if e[1][0,input] == -1:
                        break

                    word = self.vocab.idx2word[int(input)]
                    if word in e[-1]:
                        continue
                    
                    output, hidden = self.decoder(input, e[2], context) 
                    output = m(output)
                    output[0,3] = -1
                    output[0,input]=-1
                    
                    tmp.append([e[0]*float(e[1][0,input]), output, hidden,input,e[-1]+[word]])
                    e[1][0,input]=-1 
                
            ret = tmp
            ret.sort(reverse=True, key=lambda e:e[0])
            ret = ret[:int(self.beam)]
        
            
        return ret
        
        

