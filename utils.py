import time
import math
import pickle

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Corpus:
    def __init__(self, corpus_pt):
        self.corpus_pt = corpus_pt
        self.w2id = {}
        self.id2w = {}
        self.w2cnt = {}
        self.docs = []
        self.process_corpus(corpus_pt)
    
    def process_corpus(self, corpus_pt):
        for line in open(corpus_pt):
            self.process_doc(line.strip().split())
            
    def process_doc(self, words):
        doc = []
        for word in words:
            if word not in self.w2id:
                self.w2id[word] = len(self.w2id)
                self.id2w[len(self.id2w)] = word
                self.w2cnt[word] = 1
            else:
                self.w2cnt[word] += 1
            doc.append(self.w2id[word])
        self.docs.append(doc)
        
    def write_vocab(self, vocab_pt):
        output_pt = open(vocab_pt, 'w')
        for i in range(len(self.id2w)):
            output_pt.writelines(self.id2w[i] + '\n')
        output_pt.writelines('<padding>\n')
        output_pt.close()
    
    def write_docInNum(self, output_doc_pt):
        output_pt = open(output_doc_pt, 'w')
        for doc in self.docs:
            output_pt.writelines(' '.join(map(str, doc)))
            output_pt.writelines('\n')
        output_pt.close

def preprocess_data(corpus_pt, data_pt, context_size, w_freq_pt):
    data_corpus = Corpus(corpus_pt)
    #data_corpus.write_vocab(DATA_DIR + 'result/myWord2Vec/vocab.txt')
    #data_corpus.write_docInNum(DATA_DIR + 'result/myWord2Vec/docInNum.txt')
    vocab_size = len(data_corpus.w2id)
    data = []
    for docid, doc in enumerate(data_corpus.docs):
        for target_idx, target_wid in enumerate(doc):
            context_idx_lo = max(0, target_idx-context_size)
            context_idx_hi = min(len(doc), target_idx+context_size+1)
            context_wids = doc[context_idx_lo: target_idx] + doc[target_idx+1: context_idx_hi]            
            context_wids.extend([vocab_size] * (2*context_size - len(context_wids))) #padding here
            data.append((target_wid, context_wids))
    pickle.dump(data, open(data_pt, 'wb'))
    
    id2freq = [data_corpus.w2cnt[data_corpus.id2w[wid]] for wid in range(len(data_corpus.id2w))]
    pickle.dump(id2freq, open(w_freq_pt, 'wb'))
    print('preprocessing done')