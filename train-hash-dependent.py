# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import model.input as dd
import model.gensynth as gs
import numpy as np
import tensorflow as tf
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide TF debugging info

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lenseq', default=2000, type=int)
parser.add_argument('-s', '--shared', default=1000, type=int)
parser.add_argument('-b', '--batchsize', default=128, type=int)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-d', '--d_model', default=64, type=int)
parser.add_argument('-z', '--hashsz', default=64, type=int) # -h is help so we're using -z
parser.add_argument('-n', '--n_hash', default=8, type=int)
parser.add_argument('-e', '--encoder', default='chunk', type=str, choices=['chunk','graph','rnn','attn'])
parser.add_argument('-i', '--interactive', action='store_true')
parser.add_argument('--band_eval', action='store_true')
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('-t', '--target', default='/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/SRR5405830_rtrim0_final_contigs.fa', type=str) # target file or directory
parser.add_argument('-m', '--mfile', default='/fs/nexus-scratch/rhaworth/models/chunkhashdep.model.h5', type=str)
parser.add_argument('--mult', default=2.0, type=float)
parser.add_argument('--train_simple', action='store_true')
args = parser.parse_args()

chunksz = args.lenseq
min_len = args.shared
batch_size = args.batchsize
k = args.k
d_model = args.d_model
hashsz = args.hashsz
n_hash = args.n_hash # for bucketing approach
enc = args.encoder
mfile = args.mfile

# construct hashindex, extract chunks from file
# if provided dir, use all files in dir; if provided file, construct index from dir then extract file
overlap = 0.5
if os.path.isdir(args.target):
    index = dd.HashIndex(args.target, n_hash, 'fa')
    chunks = []
    for i in range(len(index.filenames)):
        chunks += index.chunks_from_file(i, chunksz, overlap, k)
else:
    index = dd.HashIndex(os.path.dirname(args.target), n_hash, 'fa')
    fileidx = index.filenames.index(args.target)
    chunks = index.chunks_from_file(fileidx, chunksz, overlap, k)

# generator helper function
def make_gen_dep(prob_sub=0.01, exp_indel_rate=0.005, exp_indel_size=10, min_len=min_len, fixed=None, simple=False):
    return gs.gen_adversarial_chunks_dependent(chunks,
                                               chunk_size=chunksz, min_shared=min_len, batch_size=batch_size,
                                               prob_sub=prob_sub, exp_indel_rate=exp_indel_rate, exp_indel_size=exp_indel_size,
                                               tokens=itokens, k=k, boundary_pad=20, min_pop=1.0,
                                               fixed=fixed, mult=args.mult, simple=simple)

itokens, _ = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
gen_train = make_gen_dep(simple=args.train_simple)

print('kmer dict size:', itokens.num())

from model.chunk_hash import ChunkMultiHash
from model.graph_hash import GraphHash
from model.rnn_hash import GRUHash

if enc == 'chunk':
    ssb = ChunkMultiHash(itokens, chunksz, d_model, hashsz, n_hash)
elif enc == 'graph':
    ssb = GraphHash(itokens, chunksz, d_model, hashsz, n_hash)
elif enc == 'rnn':
    ssb = GRUHash(itokens, chunksz, d_model, hashsz, n_hash)

def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * np.exp(-0.5)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
model_saver = tf.keras.callbacks.ModelCheckpoint(mfile, monitor='loss', save_best_only=True, save_weights_only=True)

# tensorboard
import datetime
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

ssb.compile(tf.keras.optimizers.Adam(0.001))
try: ssb.model.load_weights(mfile)
except: print('\n\nnew model')

# set verbosity
verbose = 2
if args.interactive:
    ssb.model.summary()
    verbose = 1

# train unless eval_only flag set
if not args.eval_only:
    ssb.model.fit(gen_train, steps_per_epoch=100, epochs=15, verbose=verbose,
                  #validation_data=([Xvalid, Yvalid], None), \
                  callbacks=[lr_scheduler,
                             model_saver,
                             #tensorboard_callback
                             ])
    print('done training')

# check accuracy near decision boundary
print('hard gen eval')
gen_hard = make_gen_dep()
ssb.model.evaluate(gen_hard, steps=100, verbose=verbose)

# check accuracy far from decision boundary
print('simple gen eval')
gen_simple = make_gen_dep(simple=True)
ssb.model.evaluate(gen_simple, steps=100, verbose=verbose)

# evaluate w/ fixed shared region sizes in steps of 100 if band_eval flag is set
if args.band_eval:
    print('\nband eval')
    ssb.compile(mode='eval')
    ssb.model.load_weights(mfile)
    step = 100
    for shared_len in range(step,chunksz-step+1,step):
        print('\nshared region size:', shared_len)

        # get current label
        label = int(shared_len >= min_len)

        print('hard gen eval')
        gen_hard = make_gen_dep(min_len=shared_len, fixed=label)
        ssb.model.evaluate(gen_hard, steps=100, verbose=verbose)

        # check accuracy far from decision boundary
        print('simple gen eval')
        gen_simple = make_gen_dep(min_len=shared_len, fixed=label, simple=True)
        ssb.model.evaluate(gen_simple, steps=100, verbose=verbose)