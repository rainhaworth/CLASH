# learning to hash chunks of sequences such that similar sequences tend to produce collisions
# some sections modified from https://github.com/lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py

import tensorflow as tf
import numpy as np
from model.hash_metrics import *

# COMPONENTS

# reduce continuous sequence with overlapping convs
class ConvReduceBlock(tf.keras.layers.Layer):
    def __init__(self, downsample=1024, layers=4, dim=128):
        super(ConvReduceBlock, self).__init__()
        # reduce 'downsample' kmers to 1 across N layers
        self.ds_per_layer = int(downsample ** (1 / layers))
        self.redconvs = [tf.keras.layers.Conv1D(dim, self.ds_per_layer, self.ds_per_layer, padding='valid', activation='tanh') for _ in range(layers)]
        # extra conv; reduce if needed, otherwise use simple 3-wide conv
        ds_current = self.ds_per_layer ** layers
        ds_last = downsample // ds_current
        self.ds_total = ds_current*ds_last
        self.redconvs.append(tf.keras.layers.Conv1D(dim, ds_last, ds_last, padding='valid', activation='tanh'))
        # resblocks
        self.resblocks = [ResBlock1D(dim=dim, layers=4) for _ in range(len(self.redconvs))]
    def call(self, x):
        for res, red in zip(self.resblocks, self.redconvs):
            x = res(x)
            x = red(x)
        return x

# 1D residual block
class ResBlock1D(tf.keras.layers.Layer):
    def __init__(self, dim=128, kernelsz=3, layers=2, layernorm=False, upsample=False):
        super(ResBlock1D, self).__init__()
        self.layernorm = None
        self.upsample = None
        # create convolutional layers
        self.convs = [tf.keras.layers.Conv1D(dim, kernelsz, padding='same') for _ in range(layers)]
        if layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.ReLU()
        if upsample:
            self.upsample = tf.keras.layers.Dense(dim, use_bias=False)
    def call(self, x):
        x_init = x
        if self.upsample is not None:
            x_init = self.upsample(x_init)
        # convs, intermediate relus
        for conv in self.convs[:-1]:
            x = conv(x)
            if self.layernorm is not None:
                x = self.layernorm(x)
            x = self.relu(x)
        # last conv
        x = self.convs[-1](x)
        # skip + relu
        x = x + x_init
        return self.relu(x)

# inception layer w/ dim reduction
# hybrid of original 2014 architecture and LSB architecture
class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, dim=128, maxlen=7, pool=2, layernorm=False, reduce=True):
        super(InceptionLayer, self).__init__()

        self.dim = dim
        if reduce:
            # require divisibility by # of filters
            assert (dim*pool) % (maxlen+1) == 0
            self.dim = (dim*pool) // (maxlen+1)

        self.filters = [tf.keras.layers.Conv1D(self.dim, f+1, activation='relu', padding='same')
                        for f in range(maxlen)]
        self.filters.append(tf.keras.layers.MaxPool1D(2, 1, 'same'))
        self.pool = tf.keras.layers.MaxPool1D(pool)
        #self.norm = tf.keras.layers.LayerNormalization()

        self.reducers = [None for _ in range(len(self.filters))]
        if reduce:
            # first layer does not need a reducer
            self.reducers[1:] = [tf.keras.layers.Conv1D(self.dim, 1, activation='relu', padding='same')
                                 for _ in range(len(self.filters)-1)]
    
    def call(self, x):
        # apply filters
        out = []
        for reducer, filter in zip(self.reducers, self.filters):
            o = x
            if reducer != None:
                o = reducer(o)
            o = filter(o)
            #o = self.pool(o)
            out.append(o)
        # concat feature maps
        return tf.concat(out, axis=-1) # removed pool

# MODELS

class ConvBloom:
    def __init__(self, tokens, chunksz=1024, d_model=256):
        self.tokens = tokens
        self.d_model = d_model
        self.c = chunksz
        d = d_model

        self.kmer_emb = tf.keras.layers.Embedding(tokens.num(), d) # b x l x d
        # feature extraction
        self.conv_pool = tf.keras.layers.MaxPool1D(2)
        self.conv_layers = [tf.keras.layers.Conv1D(d, 4, activation='relu', padding='same'), self.conv_pool,
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True),
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True), self.conv_pool,
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True),
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True), self.conv_pool,
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True),
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True), self.conv_pool,
                            ResBlock1D(dim=d, kernelsz=2, layernorm=True), ResBlock1D(dim=d, kernelsz=2, layernorm=True),
                            ]
        #self.conv_layers = [self.conv_pool]
        self.flatten = tf.keras.layers.Flatten()
        self.small_mlp = tf.keras.layers.Dense(64, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.2)
        # predict membership
        self.pred = tf.keras.layers.Dense(1, activation='sigmoid')

    def compile(self, optimizer='adam'):
        ins = tf.keras.layers.Input(shape=(self.c,), dtype='int32')
        
        x = self.kmer_emb(ins)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.small_mlp(x)
        pred = self.pred(x)

        def p(y_true, y_pred):
            return tf.reduce_mean(y_pred)
        def t(y_true, y_pred):
            return tf.reduce_mean(y_true)

        # create model
        self.model = tf.keras.Model(ins, pred)
        self.model.compile(optimizer,
                           tf.keras.losses.BinaryFocalCrossentropy(), # TODO: try focal 
                           [#tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall()]
                           )