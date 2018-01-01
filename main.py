import torch
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader
from utils import preprocess_data, timeSince
from sift_grams import CBOWData, SiftGram

USE_CUDA = True
TIE_EMBEDDINGS = False #better not tie
NEED_PREPROCESS = False
USE_WEIGHTS = True #better use weights for sampling
DATA_DIR = '/media/guangxu/O_o/UB/research/dataset/20newsgroups/'

corpus_pt = DATA_DIR + 'CoEmbedding/20news_min_cnt.txt'
data_pt = DATA_DIR + 'result/myWord2Vec/train.dat'
w_freq_pt = DATA_DIR + 'result/myWord2Vec/w_freq.dat'
embedding_save_pt = DATA_DIR + 'result/myWord2Vec/myCBOW.npz'
embedding_dim = 50
n_epochs = 50
context_size = 5 #one side only
n_neg = 10
batch_size = 1024
learning_rate = 0.005 #0.001 good

if __name__ == '__main__':

    if NEED_PREPROCESS:
        preprocess_data(corpus_pt, data_pt, context_size, w_freq_pt)
    
    dataset = CBOWData(data_pt, w_freq_pt)
    vocab_size = dataset.vocab_size
    wid_freq = dataset.w_freq
    losses = []
    model = SiftGram(vocab_size, embedding_dim, n_neg, wid_freq, USE_CUDA, TIE_EMBEDDINGS, USE_WEIGHTS)
    if USE_CUDA:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        print('%dth epoch...' % (epoch))
        total_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batchid, (target_wid, context_wids) in enumerate(dataloader):
            model.zero_grad()
            loss = model(target_wid, torch.stack(context_wids, dim=1))
            loss.backward()
            optimizer.step()       
            total_loss += loss.data
            
#            break
#        break
            
        losses.append(total_loss[0])
        print('time cost: ' + timeSince(start_time))
        np.savez(embedding_save_pt, C=model.o_embeddings.weight.data.cpu().numpy(), B=model.i_embeddings.weight.data.cpu().numpy())
        
    
    print(losses)
    #print(data_corpus.id2w[535])
    #print(data_corpus.w2id['sense'])
    #print(data_corpus.id2w[1])
    #print(data_corpus.w2id['atheism'])
    #print(data_corpus.id2w[data_corpus.docs[10][1]])