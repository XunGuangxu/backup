import numpy as np
import scipy.stats
DATA_DIR = '/media/guangxu/O_o/UB/research/dataset/'


#vocab_pt = DATA_DIR + '20newsgroups/CoEmbedding/vocab.txt'
##embedding_pt = DATA_DIR + '20newsgroups/CoEmbedding/word2vec/svd.npz'
#embedding_pt = DATA_DIR + '20newsgroups/result/myLDA/nmf.npz'


#vocab_pt = DATA_DIR + '20newsgroups/result/myLDA/wordmap.txt'
#embedding_pt = DATA_DIR + '20newsgroups/result/myLDA/pzw.txt'


#vocab_pt = DATA_DIR + '20newsgroups/result/myLDA/wordmap.txt'
#embedding_pt = DATA_DIR + '20newsgroups/result/myLDA/svd.npz'
#embedding_pt = DATA_DIR + '20newsgroups/result/myLDA/wsvd.npz'

vocab_pt = DATA_DIR + '20newsgroups/result/myWord2Vec/vocab.txt'
embedding_pt = DATA_DIR + '20newsgroups/result/myWord2Vec/myCBOW.npz'



test_pt = DATA_DIR + 'embeddingTestSet/ws/ws353.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/ws353_relatedness.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/ws353_similarity.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/bruni_men.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/radinsky_mturk.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/simlex_999a.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/luong_rare.txt'

w2id = {}
id2w = {}
list_our = []
list_ground = []

def read_vocab(voca_pt):
    for cnt, l in enumerate(open(voca_pt)):
        ws = l.strip().split()
        w2id[ws[-1]] = cnt
        id2w[cnt] = ws[-1]
        
def read_embeddings(embedding_pt):
    if embedding_pt.endswith('npz'):
        data = np.load(embedding_pt)
#        gamma = data['C']
        gamma = data['B'] + data['C']
        data.close()
    else:
        gamma = np.zeros((100, len(w2id)), dtype=np.float32)
        for cnt,line in enumerate(open(embedding_pt)):
            ws = line.strip().split()
            gamma[cnt] = np.array(list(map(float,ws)))
        gamma = gamma.T
    return gamma

def most_similar(dword, topn):
    wid = w2id[dword]
    wvec = gamma[wid]
    distances = np.inner(-wvec, gamma)
    most_extreme = np.argpartition(distances, topn)[:topn]
    #print(np.sort(distances.take(most_extreme)))
    return [id2w[t] for t in most_extreme.take(np.argsort(distances.take(most_extreme)))]

def cosine_distance(word1, word2):
    wid1 = w2id[word1]
    wid2 = w2id[word2]
    wvec1 = gamma[wid1]
    wvec2 = gamma[wid2]
    return np.inner(wvec1, wvec2)

read_vocab(vocab_pt)
gamma = read_embeddings(embedding_pt)
print(gamma.shape)
#normalize gamma
normss = np.linalg.norm(gamma, axis = 1, keepdims = True) #axis matters
gamma = gamma/normss

for line in open(test_pt):
    ws = line.strip().split()
    if ws[0] in w2id and ws[1] in w2id:
#        print(line + '\t' + str(cosine_distance(ws[0], ws[1])))
        list_ground.append(float(ws[2]))
        list_our.append(cosine_distance(ws[0], ws[1]))

rank_our = scipy.stats.rankdata(list_our)
rank_ground = scipy.stats.rankdata(list_ground)
#print(rank_our)
#print(rank_ground)
#print(len(rank_ground))
result = scipy.stats.spearmanr(rank_ground, rank_our)
print(result)


print(most_similar('car', 10))
print(most_similar('jesus', 10))
print(most_similar('hockey', 10))
print(cosine_distance('car', 'cars'))
print(cosine_distance('car', 'medical'))
