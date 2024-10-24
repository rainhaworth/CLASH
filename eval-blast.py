# inspect clusters
import os
import csv # for writing data
import model.input as dd
import numpy as np
import time
import pickle
from math import comb # for computing max number of hits
from collections import defaultdict
import argparse
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
# model
parser.add_argument('-l', '--lenseq', default=2000, type=int)
parser.add_argument('-k', default=4, type=int)
#parser.add_argument('-d', '--d_model', default=64, type=int)
parser.add_argument('-z', '--hashsz', default=64, type=int)
parser.add_argument('-n', '--n_hash', default=8, type=int)
#parser.add_argument('-b', '--batchsz', default=1024, type=int)
#parser.add_argument('-m', '--mfile', default='/fs/nexus-scratch/rhaworth/models/chunkhashdep.model.h5', type=str)
parser.add_argument('-r', '--resultdir', default='/fs/nexus-scratch/rhaworth/output/', type=str)
#parser.add_argument('-i', '--indir', default='/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/', type=str)
parser.add_argument('--mode', choices={'allvsall', 'srctgt'}, default='allvsall', type=str)
parser.add_argument('--srcindex', default='hashindex.pickle', type=str)
parser.add_argument('--tgtindex', default='', type=str)
parser.add_argument('--blast', default='prem-micro-blast-reduced.txt', type=str) # blast-reduce.py output file
parser.add_argument('--ext', default='fa', type=str)
parser.add_argument('-o', '--output', type=str, choices={'summary', 'allhits', 'gephicsv'}, default='summary')

args = parser.parse_args()

max_len = args.lenseq
k = args.k
n_hash = args.n_hash
results_dir = args.resultdir
dataset_path = results_dir
blast_red_file = args.blast
ext = args.ext

# load hash indices
hash_index_fn = os.path.join(dataset_path, args.srcindex)
with open(hash_index_fn, 'rb') as f:
    hash_index_src : dd.HashIndex = pickle.load(f)
print(len(hash_index_src.filenames), 'source files found')

# in all-vs-all, source and target are the same
if args.mode == 'allvsall':
    hash_index_tgt = hash_index_src
else:
    hash_index_fn = os.path.join(dataset_path, args.tgtindex)
    with open(hash_index_fn, 'rb') as f:
        hash_index_tgt : dd.HashIndex = pickle.load(f)
    print(len(hash_index_tgt.filenames), 'target files found')

# get chunk metadata
metadata_src = []
for i in range(len(hash_index_src.filenames)):
    # TODO: overlap param
    _, mds = hash_index_src.chunks_from_file(i, max_len, 0.5, k, True)
    metadata_src.append(mds)
metadata_tgt = []
for i in range(len(hash_index_tgt.filenames)):
    # TODO: overlap param
    _, mds = hash_index_tgt.chunks_from_file(i, max_len, 0.5, k, True)
    metadata_tgt.append(mds)

# pad and convert to numpy array
max_md_src = max(len(md) for md in metadata_src)
max_md_tgt = max(len(md) for md in metadata_tgt)
for i in range(len(metadata_src)):
    if len(metadata_src[i]) < max_md_src:
        metadata_src[i] += [''] * (max_md_src - len(metadata_src[i]))
for i in range(len(metadata_tgt)):
    if len(metadata_tgt[i]) < max_md_tgt:
        metadata_tgt[i] += [''] * (max_md_tgt - len(metadata_tgt[i]))
metadata_src = np.array(metadata_src)
metadata_tgt = np.array(metadata_tgt)

# make list of BLAST hits
blast_red_file = os.path.join(results_dir, blast_red_file)
blast_dict = defaultdict(list)
with open(blast_red_file, 'r') as f:
    lastquery = ''
    for line in f:
        # get metadata for this line if applicable
        if len(line) < 2:
            continue
        if 'Query=' in line:
            lastquery = line[7:].strip()
        else:
            md =' '.join(line.split()[:-2]) # remove score numbers and whitespace
            blast_dict[lastquery].append(md)

# helper function: get all hashes matching a metadata string from a given 2D metadata numpy array + hash index
def md2hash(md, md_arr, hash_index : dd.HashIndex):
    # if no matches, return none
    if md not in md_arr:
        return None, 0
    
    # iterate over indices matching md, retrieve hashes
    indices = np.argwhere(md_arr == md)
    hashes = []
    key_not_found = 0
    for data in indices:
        data = tuple(data)
        try:
            hashes.append(hash_index.get_hashes(data))
        except KeyError:
            key_not_found += 1
            continue

    if len(hashes) == 0:
        return None, 0

    # return in (hash_table, index) format
    return np.array(hashes).T, key_not_found

# validate
print('running eval')
hits = 0
misses = 0
rejected = 0
chunk_collisions = 0
seq_collisions = 0
chunk_coll_per_seq = defaultdict(int)
key_not_found = 0
for key, matches in tqdm(blast_dict.items()):
    # get hashes, reject all matches if none found
    key_hashes, knf = md2hash(key, metadata_tgt, hash_index_tgt)
    key_not_found += knf
    if key_hashes is None:
        rejected += len(matches)
        continue

    # recall: iterate over source dataset BLAST hits, find those that have been hashed and check for collisions
    # recall = hits / (hits + misses)
    for val in matches:
        val_hashes, knf = md2hash(val, metadata_src, hash_index_src)
        key_not_found += knf
        if val_hashes is None:
            rejected += 1
            continue
        
        # iterate over hash tables, check for matches with set operations
        hit = False
        for i in range(n_hash):
            if bool(set(val_hashes[i]) & set(key_hashes[i])):
                hit = True
                break
        
        # update counters
        if hit:
            hits += 1
        else:
            misses += 1

    # count unique hash collisions between source and target sequences
    # convert back to (index, hash_table) format
    key_hashes = key_hashes.T
    mds_src = set()
    for hashes in key_hashes:
        # output: list of lists
        data = hash_index_src.get_data(hashes)
        # flatten + get unique
        data = set(itertools.chain(*data))
        # count each unique chunk collision
        chunk_collisions += len(data)
        # get unique sequences
        for c in data:
            chunk_coll_per_seq[(key, metadata_src[c])] += 1
            mds_src.add(metadata_src[c])
    # count each unique sequence collision
    seq_collisions += len(mds_src)

# compute possible chunk collisions per sequence
idy_per_seq = dict()
for key, colls in chunk_coll_per_seq.items():
    total = np.count_nonzero(metadata_tgt == key[0]) * np.count_nonzero(metadata_src == key[1])
    idy_per_seq[key] = colls / total

# hardcode this for now
threshold = 0.3
over_thresh_blast_hits = 0
over_thresh_blast_misses = 0
for key, idy in idy_per_seq.items():
    if idy >= threshold:
        if key[1] in blast_dict[key[0]]:
            over_thresh_blast_hits += 1
        else:
            over_thresh_blast_misses += 1



# compute possible collisions on all hashed sequences
chunk_comparisons = 0
seq_comparisons = 0
tgt_chunks = []
tgt_seqs = set()
src_seqs = set()
for i in range(len(hash_index_tgt.filenames)):
    chunks, mds = hash_index_tgt.chunks_from_file(i, max_len, 0.5, k, metadata=True)
    tgt_chunks += chunks
    tgt_seqs = tgt_seqs.union(set(mds))
for i in range(len(hash_index_src.filenames)):
    src_chunks, mds = hash_index_src.chunks_from_file(i, max_len, 0.5, k, metadata=True)
    chunk_comparisons += len(tgt_chunks) * len(src_chunks)
    src_seqs = src_seqs.union(set(mds))
seq_comparisons = len(tgt_seqs) * len(src_seqs)

print('BLAST recall:', hits / (hits + misses))
print('BLAST hits and misses:', hits, misses)
print('BLAST pairs not found:', rejected)
print('chunk collisions:', chunk_collisions, '/', chunk_comparisons, '({}%)'.format(chunk_collisions / chunk_comparisons * 100.0))
print('seq collisions:', seq_collisions, '/', seq_comparisons, '({}%)'.format(seq_collisions / seq_comparisons * 100.0))
print('keys not found:', key_not_found)
print('new thing:', len(chunk_coll_per_seq.items()), max(chunk_coll_per_seq.values()))
print('identity stats:', len(idy_per_seq.items()), max(idy_per_seq.values()), min(idy_per_seq.values()), np.mean(list(idy_per_seq.values())), np.median(list(idy_per_seq.values())))
print('BLAST hits found, not found threshold ({}):'.format(threshold), over_thresh_blast_hits, over_thresh_blast_misses)

sns.histplot(data=list(idy_per_seq.values()))
plt.title('Identity frequencies')
plt.savefig('identity.png', dpi=200)