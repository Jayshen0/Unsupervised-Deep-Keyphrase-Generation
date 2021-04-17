import torch
import pickle
from torch.utils.data import DataLoader
from my_dataloader import *
from create_vocabulary import *
from Model import Encoder, Decoder, Seq2Seq
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#encoder = Encoder(input_dim=2999, name='emb_inspec.npy')
#decoder = Decoder(output_dim=2999, name='emb_inspec.npy')
encoder = Encoder()
decoder = Decoder()
model = Seq2Seq(encoder, decoder, device).to(device)
#model.load_state_dict(torch.load('train.pt'))

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

batch=64

tot_epoch = 100

vocab = np.load('vocab_kp20k2.npy', allow_pickle=True).item()
#vocab = np.load('vocab_inspec.npy', allow_pickle=True).item()
TRG_PAD_IDX = vocab('<pad>')
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, step_size=6, gamma=0.8)
#train_data = MyDataset(data_name='inspec2.npy', vocab_name='vocab_inspec.npy')
train_data = MyDataset()
test_data = MyDataset(cls2=1)

train_loader = DataLoader(train_data, batch_size=batch,num_workers=2,shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch,num_workers=2,shuffle=False)
#val_loader = DataLoader(val_data,  batch_size=batch,num_workers=8,shuffle=False)
#prev_tot = float('inf')

def train(iterator):
    
    model.train()
    
    epoch_loss = 0
    cnt=0
    m = 0
    for i,(src,trg) in enumerate(iterator):
    #for i,(x,cls) in enumerate(iterator):
        src = src.long().permute(1,0).to(device)
        trg = trg.long().permute(1,0).to(device)
       

        optimizer.zero_grad()
        
        output = model.forward(src, trg)
        
        #print(output.shape, trg.shape)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        
        trg = trg[1:].reshape(5*trg.shape[1])
         
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)

        
        loss.backward()
        optimizer.step()
    

        epoch_loss += loss.item()
    torch.cuda.empty_cache()
    scheduler.step()
    return epoch_loss / len(iterator)

def evaluate(iterator):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (src, trg) in enumerate(iterator):
            src = src.long().permute(1,0).to(device)
            trg = trg.long().permute(1,0).to(device)
        

            output = model.forward(src, trg) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(5*trg.shape[1])


            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


best_valid_loss = float('inf')

for epoch in range(30):
    

    train_loss = train(train_loader)
    valid_loss = evaluate(test_loader)

    #valid_loss = train_loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print('saved')
        torch.save(model.state_dict(), 'train.pt')
    
    print(epoch,':')
    print(train_loss, valid_loss)
    
    print('****************************************\n')
