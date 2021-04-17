import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self, input_dim=50004, emb_dim=200, hid_dim=256, dropout=0.5,name='emb_kp20k2.npy'):
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
    
        return hidden, outputs
    



class Decoder(nn.Module):
    def __init__(self, output_dim=50004, emb_dim=200, hid_dim=256, dropout=0.5, name='emb_kp20k2.npy'):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)


        self.attention_layer = nn.Sequential(
                             nn.Linear(self.hid_dim, self.hid_dim),
                                         nn.ReLU(inplace=True)
                                                 )
        self.rnn = nn.LSTM(emb_dim, hid_dim)
        
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
        h,c = hidden
        context = nn.Tanh()(context)
        h = self.attention_layer(h)
        w = torch.bmm(context, h.permute(1,2,0)) 
        w = w.squeeze()
        w = F.softmax(w,dim=-1) 
        #print(w.shape, context.shape)
        w = torch.bmm(w.unsqueeze(1), context)
        w = w.squeeze() 
        output = torch.cat((embedded.squeeze(0),w),
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
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    
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
        context,aa = self.encoder(src)
        
        #context also used as the initial hidden state of the decoder
        hidden = context

        h,c = hidden
        h = 0.5*(h[0][:][:].squeeze(0)+h[1][:][:].squeeze(0))
        c = 0.5*(c[0][:][:].squeeze(0)+c[1][:][:].squeeze(0))
        hidden = (h.unsqueeze(0),c.unsqueeze(0))
        
        aa = 0.5*(aa[:,:,:256]+aa[:,:,256:])
        aa = aa.permute(1,0,2)


        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
    
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, aa)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
        
        

