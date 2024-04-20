# input pipeline for tfv2 original transformer implementation
# also used for skewed cross-attention
import os, glob
import numpy as np

# custom TokenList and pad_to_longest; do not import from dataloader.py
class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>'] + token_list
        self.t2id = {v:k for k,v in enumerate(self.id2t)}
    def id(self, x):    return self.t2id.get(x, 1)
    def token(self, x):    return self.id2t[x]
    def num(self):        return len(self.id2t)
        
def pad_to_longest(xs, tokens, max_len=999):
    longest = min(len(max(xs, key=len)), max_len)
    X = np.zeros((len(xs), longest), dtype='int32')
    for i, x in enumerate(xs):
        for j, z in enumerate(x):
            X[i,j] = tokens.id(z)
    return X

# pad to max: always use max_len regardless of inputs
def pad_to_max(xs, tokens, max_len=999):
    X = np.zeros((len(xs), max_len), dtype='int32')
    for i, x in enumerate(xs):
        for j, z in enumerate(x):
            X[i,j] = tokens.id(z)
    return X

# load list of kmers from dna2vec, duplicate, return as itokens, otokens
def LoadKmerDict(dict_file=None, k=8):
    if dict_file is None or os.path.exists(dict_file) == False:
        raise FileNotFoundError(dict_file)
    
    # modified from ljqpy.LoadList()
    with open(dict_file, encoding="utf-8") as f:
        kmers = list(ll[:k] for ll in f.read().split('\n') if len(ll) >= k)
    
    # return as 2 separate TokenLists
    return TokenList(kmers), TokenList(kmers)

# generator that loads all data found in a directory
def LoadFromDir(dir, k=8, max_len=999, ext='fa'):
    # valid file extensions:
        # 'fa' = FASTA files, assumed to contain both sequences and metadata strings
        # 'txt' = text files, assumed to contain only sequences
        # if type is not in this list, handle like 'fa'
    # get list of filenames
    input_files = glob.glob(os.path.join(dir, '*.' + ext))

    for file in input_files:
        with open(file) as f:
            # loop over all lines
            metadata = None
            while True:
                instr = f.readline()
                if not instr or len(instr) < k:
                    break
                # grab metadata
                if instr[0] == '>' and ext != 'txt':
                    metadata = instr[1:]
                    continue
                # split string
                curr_seg = 0
                adj_max = max_len+k-1
                for i in range(0,len(instr), adj_max):
                    # yield duplicates
                    yield [instr[i:i+adj_max], instr[i:i+adj_max]], metadata, curr_seg
                    curr_seg += 1
                metadata = None

# build index of file data, stored in memory
# store as sequences to save memory
class DataIndex:
    def __init__(self, dir, itokens, otokens, k=8, max_len=4096, split=False, fasta=False, metadata=False):
        # split: produce separate indices for each file (not implemented)
        self.k = k
        self.itokens = itokens
        self.otokens = otokens
        self.max_len = max_len
        self.split = split
        self.mdindex = None

        if fasta:
            ext = 'fa'
        else:
            ext = 'txt'

        self.index = []
        self.seg_index = []
        if metadata:
            self.mdindex = []
        
        for ss, md, seg in LoadFromDir(dir, k, max_len, ext):
            # just store X, drop Y
            self.index.append(ss[0])
            self.seg_index.append(seg)
            if metadata and md is not None:
                self.mdindex.append(md)

    def get(self, idx, file=None):
        # for indices split across multiple files, pass in the desired file number (array index)
        if file != None:
            data = self.index[file][idx]
        else:
            data = self.index[idx]
        # generate list of kmers, fetch tokens, pad
        num_kmers = len(data) - self.k + 1
        return pad_to_max(
            [[data[i:i+self.k] for i in range(num_kmers)]],
            self.itokens, self.max_len
        )
    
    def getmd(self, idx):
        if self.mdindex is None:
            return None
        return self.mdindex[idx]
    
    def getseg(self, idx):
        return self.seg_index[idx]

    def len(self):
        if self.split == False:
            return len(self.index)
        else:
            # iterate over files
            length = 0
            for file in self.index:
                length += len(file)
            return length

if __name__ == '__main__':
    # test code
    itok, otok = LoadKmerDict('../utils/8mers.txt')
    print('loaded', itok.num(), 'tokens')