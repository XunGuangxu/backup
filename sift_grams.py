import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

class myCBOWNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_neg, wid_freq, USE_CUDA, TIE_EMBEDDINGS, USE_WEIGHTS):
        super(myCBOWNS, self).__init__()
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
    
    def forward(self, target_wids, context_wids):
        batch_size = len(target_wids)        
        var_context_wids = Variable(context_wids)
        var_target_wids = Variable(target_wids)
        if self.USE_CUDA:
            var_context_wids = var_context_wids.cuda()
            var_target_wids = var_target_wids.cuda()        
            
        if self.TIE_EMBEDDINGS:
            context_embeddings = self.o_embeddings(var_context_wids)
        else:
            context_embeddings = self.i_embeddings(var_context_wids)
        avg_ctxt_embeddings = context_embeddings.mean(dim=1).unsqueeze(2)
        target_embeddings = self.o_embeddings(var_target_wids).unsqueeze(1)
        if self.USE_WEIGHTS:
            var_neg_wids = Variable(torch.multinomial(self.sampling_weights, batch_size*self.n_neg, replacement=True).view(batch_size, -1))
        else:
            var_neg_wids = Variable(torch.FloatTensor(batch_size, self.n_neg).uniform_(0, self.vocab_size-1).long())
        if self.USE_CUDA:
            var_neg_wids = var_neg_wids.cuda()
        neg_embeddings = self.o_embeddings(var_neg_wids)
        
        pos_loss = torch.bmm(target_embeddings, avg_ctxt_embeddings).sigmoid().log().sum()
        neg_loss = torch.bmm(neg_embeddings.neg(), avg_ctxt_embeddings).sigmoid().log().sum()
        
        return -(pos_loss + neg_loss)
    
    
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
