# train tfv2 transformer with kmer data
# modified from en2de_main.py and pinyin_main.py
import os, sys
import model.input as dd
import model.gensynth as gs
import numpy as np
import tensorflow as tf
import argparse

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide TF debugging info

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lenseq', default=2000, type=int)
parser.add_argument('-b', '--batchsize', default=128, type=int)
parser.add_argument('-k', default=4, type=int)
parser.add_argument('-d', '--d_model', default=64, type=int)
parser.add_argument('-e', '--encoder', default='chunk', type=str, choices=['chunk','graph','rnn','attn'])
parser.add_argument('-i', '--interactive', action='store_true')
parser.add_argument('--band_eval', action='store_true')
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('-t', '--target', default='/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/SRR5405830_rtrim0_final_contigs.fa', type=str) # target file or directory
parser.add_argument('-m', '--mfile', default='/fs/nexus-scratch/rhaworth/models/bloom.model.h5', type=str)
parser.add_argument('--mult', default=2.0, type=float)
parser.add_argument('--train_simple', action='store_true')
args = parser.parse_args()

chunksz = args.lenseq
batch_size = args.batchsize
k = args.k
d_model = args.d_model
enc = args.encoder
mfile = args.mfile

# weh adding this so i don't have to do more work
n_hash = 1

# set overlap so step = int(chunksz*(1-overlap)) = 1
# -1e-9 at the end to be sure we won't go below 1 due to rounding error
overlap = 1.0 - 1.0/chunksz - 1e-9

# construct hashindex, extract chunks from file
# if provided dir, use all files in dir; if provided file, construct index from dir then extract file
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
def make_gen(prob_sub=0.01, exp_indel_rate=0.005, exp_indel_size=10):
    return gs.gen_chunks_bloom(chunks, itokens,
                                chunk_size=chunksz, batch_size=batch_size,
                                prob_sub=prob_sub, exp_indel_rate=exp_indel_rate, exp_indel_size=exp_indel_size,
                                k=k, mult=args.mult)

itokens, _ = dd.LoadKmerDict('./utils/' + str(k) + 'mers.txt', k=k)
gen_train = make_gen()

# compute total # batches to iterate over dataset then aim to complete iteration over dataset in 5 epochs
# if you don't want this behavior just use steps_per_epoch=100
batches_per_dataset = len(chunks) // batch_size
steps_per_epoch = batches_per_dataset // 5

print('chunk count:', len(chunks))

print('kmer dict size:', itokens.num())

from model.bloom import ConvBloom

# TODO: other encoders
ssb = ConvBloom(itokens, chunksz, d_model)

def lr_schedule(epoch, lr):
    if epoch < 3:
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
    ssb.model.fit(gen_train, steps_per_epoch=steps_per_epoch, epochs=5, verbose=verbose,
                  #validation_data=([Xvalid, Yvalid], None), \
                  callbacks=[lr_scheduler,
                             #model_saver,
                             #tensorboard_callback
                             ])
    print('done training')

# check accuracy
print('eval')
gen_hard = make_gen()
ssb.model.evaluate(gen_hard, steps=100, verbose=verbose)