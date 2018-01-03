import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

class SiftGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_neg, wid_freq, USE_CUDA, TIE_EMBEDDINGS, USE_WEIGHTS):
        super(SiftGram, self).__init__()
        self.i_embeddings = nn.Embedding(vocab_size + 1, embedding_dim) # one more for padding
        self.o_embeddings = nn.Embedding(vocab_size + 1, embedding_dim) # one more for padding
#        self.embeddings.weight = nn.Parameter(torch.FloatTensor(vocab_size+1, embedding_dim).uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim))        
        self.vocab_size = vocab_size
        self.n_neg = n_neg
        wf = np.array(wid_freq)
        wf = wf / wf.sum()
        wf = np.power(wf, 0.75)
        self.sampling_weights = torch.FloatTensor(wf)
        self.USE_CUDA = USE_CUDA
        self.TIE_EMBEDDINGS = TIE_EMBEDDINGS
        self.USE_WEIGHTS = USE_WEIGHTS
        
        self.attn = Attention('self_con', embedding_dim, self.n_neg)
    
    def forward(self, target_wids, context_wids):
        batch_size = len(target_wids)
        
        var_context_wids = Variable(context_wids)
        var_target_wids = Variable(target_wids)
        if self.USE_WEIGHTS:
            var_neg_wids = Variable(torch.multinomial(self.sampling_weights, batch_size*self.n_neg, replacement=True).view(batch_size, -1))
        else:
            var_neg_wids = Variable(torch.FloatTensor(batch_size, self.n_neg).uniform_(0, self.vocab_size-1).long())
        if self.USE_CUDA:
            var_context_wids = var_context_wids.cuda() #batch_size * context_size
            var_target_wids = var_target_wids.cuda() #batch_size
            var_neg_wids = var_neg_wids.cuda() #batch_size
            
#        print(var_context_wids.size(), var_target_wids.size(), var_neg_wids.size())
            
        

        other_context_embeddings = self.o_embeddings(var_context_wids) #batch_size * context_size * embed_dim
        context_embeddings = self.i_embeddings(var_context_wids) #batch_size * context_size * embed_dim
        target_embeddings = self.o_embeddings(var_target_wids).unsqueeze(1) #batch_size * 1 * embed_dim
        neg_embeddings = self.o_embeddings(var_neg_wids) #batch_size * n_neg * embed_dim
        
        use_attn = random.random() < 10.5
        if use_attn:
#           print(context_embeddings.size(), avg_ctxt_embeddings.size(), target_embeddings.size(), neg_embeddings.size())
            attn_weights = self.attn(batch_size, target_embeddings, context_embeddings, other_context_embeddings) #batch_size * 1 * context_size
            attn_ctxt_embeddings = torch.bmm(attn_weights, context_embeddings).view(batch_size, -1, 1) #batch_size * embed_dim * 1
#           print(attn_weights.size(), attn_ctxt_embeddings.size())
            pos_loss = torch.bmm(target_embeddings, attn_ctxt_embeddings).sigmoid().log().sum()
            neg_loss = torch.bmm(neg_embeddings.neg(), attn_ctxt_embeddings).sigmoid().log().sum()
        
        else:
            avg_ctxt_embeddings = context_embeddings.mean(dim=1).unsqueeze(2) #batch_size * embed_dim * 1
        
            pos_loss = torch.bmm(target_embeddings, avg_ctxt_embeddings).sigmoid().log().sum()
            neg_loss = torch.bmm(neg_embeddings.neg(), avg_ctxt_embeddings).sigmoid().log().sum()
        
        return -(pos_loss + neg_loss)
    

class Attention(nn.Module):
    def __init__(self, mode, embedding_dim, n_neg, context_size=10):
        super(Attention, self).__init__()
        self.mode = mode
        
        if self.mode == 'self_tar':
            self.layer1 = nn.Linear(embedding_dim, 20)
            self.layer2 = nn.Linear(20, context_size)
        
        if self.mode == 'self_con':
            self.layer1 = nn.Linear(embedding_dim * context_size, 50)
            self.layer2 = nn.Linear(50, context_size)
        
        if self.mode == 'mutual_gen':
            self.layer1 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, batch_size, target_embeddings, context_embeddings, other_context_embeddings):
#        print('this')
#        print(target_embeddings.size(), other_context_embeddings.size())
        if self.mode == 'self_tar':
            x = F.tanh(self.layer1(target_embeddings))
            x = F.softmax(self.layer2(x))
            print(x[1], x[1].sum())
            print('normalization dimension wrong here')
            return x
        if self.mode == 'self_con':
            x = F.tanh(self.layer1(context_embeddings.view(batch_size, -1)))
            x = F.softmax(self.layer2(x))
            return x.unsqueeze(1)
        if self.mode == 'mutual_dot':
            x = torch.bmm(target_embeddings, torch.transpose(other_context_embeddings, 1, 2)).squeeze()
            x =  F.softmax(x)
#            print(x.unsqueeze(1)[1], x.unsqueeze(1)[1].sum())
            return x.unsqueeze(1)
        if self.mode == 'mutual_gen':
            x = self.layer1(target_embeddings)
#            print(x.size())
            x = torch.bmm(x, torch.transpose(other_context_embeddings, 1, 2)).squeeze()
            x =  F.softmax(x)
#            print(x.unsqueeze(1)[1], x.unsqueeze(1)[1].sum())
            return x.unsqueeze(1)

    
class CBOWData(Dataset):
    def __init__(self, data_pt, w_freq_pt):
        self.data = pickle.load(open(data_pt, 'rb'))
        self.w_freq = pickle.load(open(w_freq_pt, 'rb'))
        self.vocab_size = len(self.w_freq)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target_wid, context_wids = self.data[idx]
        return target_wid, context_wids
