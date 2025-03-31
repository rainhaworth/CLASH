# define data generator
# requires tensorflow, so kept separate from input.py
import numpy as np
import random
from tensorflow import constant, expand_dims, ragged
from .input import pad_to_max, pad_to_min_chunk

# utility functions
def gen_seq(length, lst=False):
    seq_lst = np.random.choice(['A','C','G','T'], size=length)
    if not lst:
        return ''.join(seq_lst)
    return seq_lst
def seq2kmers(seq, k):
    num_kmers = len(seq) - k + 1
    return [seq[i:i+k] for i in range(num_kmers)]

def make_ragged(xs, tokens):
    X = []
    for i, x in enumerate(xs):
        X.append([])
        for j, z in enumerate(x):
            X[i].append(tokens.id(z))
    return ragged.constant(X)

# generate pairs of data with shared blocks
# output = list of padded kmer lists
# for binary classification
def gen_simple_block_data_binary(max_len=4096, min_len=500, block_max=None, block_min=None, batch_size=8, tokens=None, k=8):
    """Generate contrastive samples

    Parameters:
        max_len, min_len: bounds for total sequence length
        block_max, block_min: bounds for block length
        batch_size: fixed batch size to yield
        tokens: tokens to represent kmers; leave as None to produce strings (not yet implemented)
        k: length of kmer
    Returns:
        A list of strings or a list of lists of kmers
    """
    seqs = [[],[]]
    labels = []

    # adjust max_len
    max_len = max_len + k - 1

    assert batch_size % 2 == 0

    # set parameters
    if block_max is None or block_max > max_len:
        block_max = max_len
    if block_min is None:
        block_min = min_len
    elif block_min > max_len:
        block_min = max_len

    # temporarily added random max_len per batch, sampled from hardcoded list of valid options
    #max_lens = [4500, 7500, 15000]
    max_len_init = max_len
    min_chunk_pop = 1.0 # fraction of chunk that must be populated
    chunksz = 2000 # hardcoded

    # return integer: additional # of nucleotides to generate to meet chunk population bound
    def round_to_pop(seqlen, pop, chunk):
        if seqlen % chunk < chunk * pop and seqlen % chunk != 0:
            return int(np.ceil(chunk * pop) - (seqlen % chunk))
        return 0
        
    while True:
        #max_len = np.random.randint(min_len*2, max_len_init)
        block_max = max_len
        # ensure labels are evenly balanced
        sample_labels = np.concatenate([np.ones(batch_size//2), np.zeros(batch_size//2)])
        np.random.shuffle(sample_labels)

        block_lengths = np.random.randint(block_min, block_max-1, size=batch_size)
        for idx in range(batch_size):
            block_length = block_lengths[idx]
            block = gen_seq(block_length)

            if sample_labels[idx] == 1:
                # generate positive labeled sequences
                for i in range(2):
                    # make them at least large enough to hold the block
                    len_seq = max_len - block_length #random.randint(max(block_length, min_len), max_len-1) # min_len -> max_len-k-block_length

                    # ensure last chunk population bound is satisfied
                    #full_len = len_seq + block_length
                    #len_seq += round_to_pop(full_len, min_chunk_pop, chunksz)

                    if max_len == block_length: # len_seq -> max_len
                        seqs[i].append(seq2kmers(block, k))
                        continue
                    seq = gen_seq(max_len - block_length) # len_seq -> max_len
                    # insert block at random point
                    if len(seq) < 2:
                        insert_point = 0
                    else:
                        insert_point = random.randint(0, len(seq)-1)
                    seq = seq[:insert_point] + block + seq[insert_point:]
                    seq = seq[:max_len]
                    seqs[i].append(seq2kmers(seq, k))
                labels.append(1)
            else:
                # generate random negative labeled sequences; ignore minimum length
                #seq1_len = random.randint(min_len, max_len-1) # for ChunkHash, replaced k -> min_len
                #seq2_len = random.randint(min_len, max_len-1)
                #seq1_len += round_to_pop(seq1_len, min_chunk_pop, chunksz)
                #seq2_len += round_to_pop(seq2_len, min_chunk_pop, chunksz)
                randseq1 = gen_seq(max_len) # seq1/2_len -> max_len
                randseq2 = gen_seq(max_len)
                seqs[0].append(seq2kmers(randseq1, k))
                seqs[1].append(seq2kmers(randseq2, k))
                labels.append(0)

        # yield complete list of sequences
        #a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        #a, b = make_ragged(seqs[0], tokens), make_ragged(seqs[1], tokens)
        longest_a = len(max(seqs[0], key=len))
        longest_b = len(max(seqs[1], key=len))
        longest = max(longest_a, longest_b)
        a, b = pad_to_min_chunk(seqs[0], tokens, longest, chunksz), pad_to_min_chunk(seqs[1], tokens, longest, chunksz)
        yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []


# adversarial block data
# generates data near the decision boundary
def gen_adversarial_block_data_binary(max_len=4096, min_len=500, block_max=None, block_min=None,
                                      prob_sub=0.05, exp_indel=1, batch_size=8, tokens=None, k=8):
    """Generate contrastive samples

    Parameters:
        max_len, min_len: bounds for total sequence length
        block_max, block_min: bounds for block length
        prob_sub: probability of random substitution events and insertion/deletion events
        batch_size: fixed batch size to yield
        tokens: tokens to represent kmers; leave as None to produce strings (not yet implemented)
        k: length of kmer
    Returns:
        A list of strings or a list of lists of kmers
    """

    boundary_pad = 100
    
    seqs = [[],[]]
    labels = []

    assert batch_size % 2 == 0

    # set parameters
    if block_max is None or block_max > max_len:
        block_max = max_len
    if block_min is None:
        block_min = min_len
    elif block_min > max_len:
        block_min = max_len

    while True:
        # pre-generate random number batches
        block_lengths_T = np.random.randint(block_min, block_max-1, size=batch_size//2)
        block_lengths_F = np.random.randint(k, block_min-boundary_pad,  size=batch_size//2)

        indel_counts = np.random.poisson(exp_indel, size=batch_size)
        ins_fracs = np.random.uniform(size=batch_size)

        for idx in range(batch_size):
            # alternate between true and false samples
            label = idx % 2
            labels.append(label)

            # generate conserved block
            if label == 1:
                block_length = block_lengths_T[idx // 2]
            else:
                block_length = block_lengths_F[idx // 2]
            block = gen_seq(block_length)

            # generate sequences: perturb block then insert into new sequence
            for seqnum in range(2):
                # DELETIONS

                # compute number of insertions and deletions to perform
                count_ins = round(ins_fracs[idx] * indel_counts[idx])
                count_del = indel_counts[idx] - count_ins

                if count_del > 0:
                    # compute total number of nucleotides to delete
                    if label == 0:
                        if count_del >= block_length - k:
                            size_del = block_length - k
                        else:
                            size_del = np.random.randint(count_del, block_length-k)
                    elif block_length - block_min <= count_del:
                        size_del = block_length - block_min
                    else:
                        size_del = np.random.randint(count_del, block_length - block_min)
                    # for each individual deletion event (slightly smaller due to truncation)
                    sizes_del = (np.random.dirichlet(np.ones(count_del)) * size_del).astype(int)

                    # simulate deletion events
                    for dsz in sizes_del:
                        if len(block) - dsz <= 0:
                            block = block[:k]
                            continue
                        pos_del = np.random.randint(0, len(block) - dsz)
                        block = block[:pos_del] + block[pos_del+dsz:]
                
                # SUBSTITUTIONS
                
                # compute number of random substitutions to perform
                count_sub = np.random.binomial(len(block), prob_sub)
                if label == 1:
                    count_sub = min(count_sub, len(block) - block_min)
                else:
                    count_sub = min(count_sub, len(block) - k)

                # simulate substitution events
                if count_sub > 0:
                    seq_sub = gen_seq(count_sub)
                    idx_sub = sorted(random.sample(range(len(block)), count_sub))

                    newblock = block[0:idx_sub[0]]
                    for i in range(count_sub-1):
                        newblock += seq_sub[i]
                        newblock += block[idx_sub[i]+1:idx_sub[i+1]]
                    newblock += seq_sub[-1]
                    if idx_sub[-1]+1 < len(block):
                        newblock += block[idx_sub[-1]+1:]
                    
                    block = newblock

                # ASSEMBLY AND INSERTIONS
                
                # if block is huge, skip all of this
                if max_len - len(block) <= k:
                    seqs[seqnum].append(seq2kmers(block, k))
                    continue
                
                # generate random part of seq
                seq_rand = gen_seq(np.random.randint(k, max_len-len(block)))

                # choose (count_ins+1) insertion points; model insertion by breaking into fragments
                if count_ins+1 > min(len(seq_rand), len(block)):
                    count_ins = min(len(seq_rand), len(block))-1

                rand_ins_idxs = sorted(random.sample(range(len(seq_rand)), count_ins+1))
                pos_block_ins = sorted(random.sample(range(len(block)), count_ins))
                pos_block_ins.insert(0, 0) # set first insertion point to 0

                # assemble sequence
                seq = seq_rand[0:rand_ins_idxs[0]]
                for i in range(count_ins):
                    seq += block[pos_block_ins[i]:pos_block_ins[i+1]]
                    seq += seq_rand[rand_ins_idxs[i]:rand_ins_idxs[i+1]]
                seq += block[pos_block_ins[-1]:]
                seq += seq_rand[rand_ins_idxs[-1]:]

                seqs[seqnum].append(seq2kmers(seq, k))

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, max_len), pad_to_max(seqs[1], tokens, max_len)
        yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []

# helper function: simulate evolution on a single sequence after sampling from indel RVs
def sim_evo(seq, prob_sub, indel_count, ins_count, indel_sizes):
    # DELETIONS
    # insertions are listed first in indel_sizes, so skip them
    for i in range(ins_count, indel_count):
        # get size
        dsz = indel_sizes[i]
        
        # if we would delete more than the full sequence, proceed with empty sequence
        if len(seq) - dsz <= 0:
            seq = ''
            break

        # delete random chunk
        pos_del = np.random.randint(0, len(seq)-dsz)
        seq = seq[:pos_del] + seq[pos_del+dsz:] # type: ignore
    
    # SUBSTITUTIONS
    if len(seq) > 0:
        seq = list(seq)
        # compute number of random substitutions to perform
        count_sub = np.random.binomial(len(seq), prob_sub)

        # simulate substitution events
        if count_sub > 0:
            seq_sub = gen_seq(count_sub)
            idx_sub = random.sample(range(len(seq)), count_sub)

            for j, i in enumerate(idx_sub):
                seq[i] = seq_sub[j]
    else:
        # guarantee seq is a list
        seq = []

    # INSERTIONS
    for i in range(ins_count):
        # generate new sequence and insert at random position
        isz = indel_sizes[i]
        ins_seq = list(gen_seq(isz, lst=True))

        if len(seq) == 0:
            seq = ins_seq
        else:
            ins_pos = np.random.randint(0, len(seq))
            # unfortunately this is the best way to insert multiple list elements in python
            seq[ins_pos:ins_pos] = ins_seq

    # convert back to string and return
    return ''.join(seq)

# helper function: sample RVs for sim_evo indels
def sample_indel_RVs(shared_length, exp_indel_rate, exp_indel_size):
    exp_indel_count = exp_indel_rate * shared_length
    indel_count = np.random.poisson(exp_indel_count)
    ins_frac = np.random.uniform()
    indel_sizes = np.random.poisson(exp_indel_size, size=indel_count)

    ins_count = np.int32(np.round(indel_count * ins_frac))
    
    return indel_count, ins_count, indel_sizes

# helper function: enforce len(seq) == length via trimming or padding with random DNA
def enforce_length(seq, length, min_pop=1.0):
    if len(seq) > length:
        # if too long, trim
        trim_start = np.random.randint(0, len(seq) - length)
        seq = seq[trim_start : trim_start+length]
    else:
        # else, generate rest of sequence as new random DNA
        # if min_pop < 1.0, randomize sequence length in [min_pop, 1.0]
        max_seq_len = max(length - len(seq), 0)
        min_seq_len = max(int(length*min_pop) - len(seq), 0)
        if min_seq_len >= max_seq_len:
            seq_len = max_seq_len
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        # generate if necessary
        if seq_len != 0:
            # insert shared at random position
            newseq = gen_seq(seq_len)
            breakp = np.random.randint(0, len(newseq))
            seq = newseq[:breakp] + seq + newseq[breakp:]

    return seq

# alt training regime: adversarial single chunks
def gen_adversarial_chunks_binary(chunk_size=1024, min_pop=0.8, min_shared=512, boundary_pad=50,
                                  prob_sub=0.01, exp_indel_rate=0.005, exp_indel_size=10, 
                                  batch_size=32, tokens=None, k=4, fixed=None,
                                  mult=2.0, simple=False):
    # TODO: real docstring
    # fixed: set to 1 or 0 for each sample to have a given label
    seqs = [[],[]]
    labels = []

    while True:
        # pre-generate random number batches
        # if fixed, enforce shared region size; T and F are not real in this case, just makes implementation easier
        if fixed is None:
            shared_T = np.random.randint(min_shared, chunk_size-1, size=batch_size//2)
            if simple:
                shared_F = np.random.randint(k, min_shared-boundary_pad, size=batch_size//2)
            # new approach: full spectrum of shared region lengths
            else:
                shared_F = np.random.randint(k, chunk_size-1, size=batch_size//2)
        else:
            shared_T = [min_shared] * (batch_size // 2)
            shared_F = [min_shared] * (batch_size // 2)

        for idx in range(batch_size):
            # alternate between true and false samples
            label = idx % 2

            # if fixed is set, use it as the final label
            if fixed is None:
                labels.append(label)
            else:
                labels.append(fixed)

            # generate conserved region
            if label == 1:
                shared_length = shared_T[idx // 2]
            else:
                shared_length = shared_F[idx // 2]
            shared = gen_seq(shared_length)

            # multiply negative sample evo rate by mult
            if fixed:
                label_mult = 1.0
            else:
                label_mult = float(mult + label - label*mult) # result = mult when label=0, 1 when label=1

            # sample indel RVs
            if simple:
                indel_count = 0
                ins_count = 0
                indel_sizes = []
            else:
                indel_count, ins_count, indel_sizes = sample_indel_RVs(shared_length, exp_indel_rate, exp_indel_size)

            # generate sequences: perturb then insert into new sequence
            for seqnum in range(2):
                # evolve shared region
                shared = sim_evo(shared, prob_sub, indel_count, ins_count, indel_sizes)

                # trim or pad to meet chunk_size; use c + k - 1 because we want c kmers, not c bp
                seqlen = chunk_size + k - 1
                seq = enforce_length(shared, seqlen)

                # convert to kmers
                seqs[seqnum].append(seq2kmers(seq, k))

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, chunk_size), pad_to_max(seqs[1], tokens, chunk_size)
        # force overfitting
        for _ in range(1):
            yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []

# variant for dependent training; takes list of chunks as input
# TODO: merge with above
def gen_adversarial_chunks_dependent(chunks,
                                     chunk_size=1024, min_pop=0.8, min_shared=512, boundary_pad=50,
                                     prob_sub=0.01, exp_indel_rate=0.005, exp_indel_size=10, 
                                     batch_size=32, tokens=None, k=4, fixed=None,
                                     mult=2.0, simple=False):
    # TODO: real docstring
    # fixed: set to 1 or 0 for each sample to have a given label
    seqs = [[],[]]
    labels = []

    # maintain list of current chunks
    chunk_count = len(chunks)
    chunk_idxs = list(range(chunk_count))

    seqlen = chunk_size + k - 1
    seqs_idxs = [0,1]

    while True:
        # pre-generate random number batches
        # pick chunks to use, remove from chunk_idxs
        # first, handle case where we have fewer chunks remaining than batch size
        batch_idxs = []
        to_sample = batch_size
        if len(chunk_idxs) < batch_size:
            batch_idxs = chunk_idxs
            chunk_idxs = list(range(chunk_count))
            to_sample = batch_size - len(batch_idxs)
        # otherwise, sample then remove via set operation
        _batch_idxs = list(np.random.choice(chunk_idxs, to_sample, replace=False))
        chunk_idxs = list(set(chunk_idxs).difference(set(_batch_idxs)))
        batch_idxs += _batch_idxs

        # if fixed, enforce shared region size; T and F are not real in this case, just makes implementation easier
        if fixed is None:
            shared_T = np.random.randint(min_shared, chunk_size-1, size=batch_size//2)
            if simple:
                shared_F = np.random.randint(k, min_shared-boundary_pad, size=batch_size//2)
            # new approach: full spectrum of shared region lengths
            else:
                shared_F = np.random.randint(k, chunk_size-1, size=batch_size//2)
        else:
            shared_T = [min_shared] * (batch_size // 2)
            shared_F = [min_shared] * (batch_size // 2)

        for idx in range(batch_size):
            # get current chunk, store randomly in seqs
            chunk_idx = batch_idxs[idx]
            chunk = chunks[chunk_idx]
            np.random.shuffle(seqs_idxs)
            seqs[seqs_idxs[0]].append(seq2kmers(chunk, k))

            # alternate between true and false samples
            label = idx % 2

            # if fixed is set, use it as the final label
            if fixed is None:
                labels.append(label)
            else:
                labels.append(fixed)

            # sample conserved region from chunk
            if label == 1:
                shared_length = shared_T[idx // 2]
            else:
                shared_length = shared_F[idx // 2]
            shared_start = np.random.randint(0, seqlen-shared_length)
            shared = chunk[shared_start : shared_start + shared_length]

            # multiply negative sample evo rate by mult
            if fixed:
                label_mult = 1.0
            else:
                label_mult = float(mult + label - label*mult) # result = mult when label=0, 1 when label=1

            # sample indel RVs
            if simple:
                indel_count = 0
                ins_count = 0
                indel_sizes = []
            else:
                indel_count, ins_count, indel_sizes = sample_indel_RVs(shared_length,
                                                                        exp_indel_rate * label_mult,
                                                                        exp_indel_size * label_mult)

            # evolve shared region
            shared = sim_evo(shared, prob_sub*label_mult, indel_count, ins_count, indel_sizes)

            # trim or pad to meet chunk_size; use c + k - 1 because we want c kmers, not c bp
            seq = enforce_length(shared, seqlen)

            # convert to kmers
            seqs[seqs_idxs[1]].append(seq2kmers(seq, k))

        # yield complete list of sequences
        a, b = pad_to_max(seqs[0], tokens, chunk_size), pad_to_max(seqs[1], tokens, chunk_size)
        # force overfitting
        for _ in range(1):
            yield [a,b], expand_dims(constant(labels), axis=-1) # add dim to fix shape when training
        seqs = [[],[]]
        labels = []

# bloom filter: generate chunks within some evolutionary distance of canonical chunks
def gen_chunks_bloom(chunks, tokens=None,
                        chunk_size=1024,
                        prob_sub=0.01, exp_indel_rate=0.005, exp_indel_size=10, 
                        batch_size=32, k=4,
                        mult=2.0):
    seqs = []
    labels = []

    # maintain list of current chunks
    chunk_count = len(chunks)
    chunk_idxs = list(range(chunk_count))

    seqlen = chunk_size + k - 1

    while True:
        # pick chunks to use
        batch_idxs = []
        to_sample = batch_size
        # handle case where we have fewer chunks remaining than batch size
        if len(chunk_idxs) < batch_size:
            batch_idxs = chunk_idxs
            chunk_idxs = list(range(chunk_count))
            to_sample = batch_size - len(batch_idxs)
        # otherwise, sample then remove via set operation
        _batch_idxs = list(np.random.choice(chunk_idxs, to_sample, replace=False))
        chunk_idxs = list(set(chunk_idxs).difference(set(_batch_idxs)))
        batch_idxs += _batch_idxs

        for idx in range(batch_size):
            # get current chunk
            chunk_idx = batch_idxs[idx]
            chunk = chunks[chunk_idx]

            # alternate between true and false samples
            label = idx % 2
            labels.append(label)

            label_mult = 1.0
            # multiply evolutionary distance by mult for negative samples
            label_mult = float(mult + label - label*mult)

            # sample RVs
            indel_count, ins_count, indel_sizes = sample_indel_RVs(seqlen, 
                                                                    exp_indel_rate * label_mult,
                                                                    exp_indel_size * label_mult)

            # simulate evolution
            #if label == 1:
            seq = sim_evo(chunk, prob_sub*label_mult, indel_count, ins_count, indel_sizes)
            #else:
            #    seq = gen_seq(seqlen)

            # trim or pad to meet chunk_size
            seq = enforce_length(seq, seqlen)

            # convert to kmers
            seqs.append(seq2kmers(seq, k))
        
        # finalize batch
        yield pad_to_max(seqs, tokens, chunk_size), constant(labels)
        seqs = []
        labels = []

# faster variant using fortran sim_evo
# NOTE: must use sim_evo.seq2kmerids as tokenizer during inference
import model.sim_evo as se
def gen_chunks_bloom_fast(chunks, tokens=None,
                        chunk_size=1024,
                        prob_sub=0.01, exp_indel_rate=0.005, exp_indel_size=10, 
                        batch_size=32, k=4,
                        mult=2.0):
    seqs = np.zeros((batch_size, chunk_size))
    labels = np.zeros(batch_size)

    # maintain list of current chunks
    chunk_count = len(chunks)
    chunk_idxs = list(range(chunk_count))

    seqlen = chunk_size + k - 1

    while True:
        # pick chunks to use
        batch_idxs = []
        to_sample = batch_size
        # handle case where we have fewer chunks remaining than batch size
        if len(chunk_idxs) < batch_size:
            batch_idxs = chunk_idxs
            chunk_idxs = list(range(chunk_count))
            to_sample = batch_size - len(batch_idxs)
        # otherwise, sample then remove via set operation
        _batch_idxs = list(np.random.choice(chunk_idxs, to_sample, replace=False))
        chunk_idxs = list(set(chunk_idxs).difference(set(_batch_idxs)))
        batch_idxs += _batch_idxs

        for idx in range(batch_size):
            # get current chunk
            chunk_idx = batch_idxs[idx]
            chunk = chunks[chunk_idx]

            # alternate between true and false samples
            label = idx % 2
            labels[idx] = label

            label_mult = 1.0
            # multiply evolutionary distance by mult for negative samples
            label_mult = float(mult + label - label*mult)

            # sample RVs
            indel_count, ins_count, indel_sizes = sample_indel_RVs(seqlen, 
                                                                    exp_indel_rate * label_mult,
                                                                    exp_indel_size * label_mult)

            # simulate evolution
            #print(idx, chunk[:10])
            seq = se.sim_evo(chunk, seqlen, prob_sub*label_mult,
                                    indel_count, ins_count, indel_sizes)

            #print(seq[:10])
            # convert to kmers
            seqs[idx,:] = se.seq2kmerids(seq, seqlen, k)
        
        # finalize batch
        yield seqs, labels