import numpy as np
import math
import tensorflow as tf


class PMK(object):
    
    def __init__(self, rng, quant=None, shift=True, min_n=0.0, sparse=True):
        self.rng = rng
        if quant == None:
            self.quant = [32, 24, 18]
        else:
            self.quant = quant
        
        
        if shift == False:
            self.shift = [0.0] * len(self.quant)
        else:
            self.shift = []
            for q in self.quant:
                self.shift.append(np.random.random() * self.rng/q)
            
        self.min_n = 0.0
        
        if len(self.shift) != len(self.quant):
            raise ValueError
        
        self.true_shift = []
        self.adj_rng = []
        
        for q, s in zip(self.quant, self.shift):
            self.true_shift.append(rng/q * s)
            self.adj_rng.append(rng + rng/q*s)
        
        self.sparse = sparse
    
    def _encode(self, X, quant, shift, rng):
        layers = []
        while quant >= 1:
            bin_locations = tf.cast(tf.math.floor((X + shift - self.min_n) * quant/rng),tf.int64)
            dense_shape = tf.convert_to_tensor([quant] * X.shape[1], dtype=tf.int64) 
            values = tf.ones(X.shape[0], dtype=tf.float32)
            sparse_vec = tf.sparse.SparseTensor(indices=bin_locations, 
                                      values=values, dense_shape=dense_shape)
            layers.append(sparse_vec)
            quant = math.floor(quant/2)
        
        return layers
    
    def pyramid_encode(self, X, sparse=True):
        pyramid = []
        for i, (q, s, r) in enumerate(zip(self.quant, self.true_shift, self.adj_rng)):
            pyramid.append(self._encode(X, q, s, r))
        
        return pyramid
        
    
    def single_pyramid_matching(self, layers_1, layers_2, rng, quant, weight="max"):
        scores = 0.0
        #matches = [tf.Variable(0.0, name="start_match")]
        matches = [0.0]
        for i, (l1, l2) in enumerate(zip(layers_1, layers_2)):
            if weight == 'max':
                layer_weight = math.sqrt(math.pow(self.rng/quant, 2) * len(l1.shape))
            elif weight == "min":
                layer_weight = math.sqrt(math.pow(self.rng/(quant*4.0), 2))
            elif weight == "mean":
                layer_weight = math.sqrt(math.pow(self.rng/(quant*2.0), 2))
            elif weight == "flat":
                layer_weight = self.rng/quant
                
            
            #mt = tf.stack([l1, l2])
            match_count = tf.sparse.reduce_sum(tf.sparse.minimum(l1, l2))
            matches.append(match_count)
            scores += layer_weight * (match_count - matches[i])
            quant = math.floor(quant/2)
        
        return scores
    
    def pmk_dist(self, pyramid_a, pyramid_b, weight="max", norm=True):
        dist = tf.Variable(0.0, name="dist")
        dist = 0.0
        for i, (layers_a, layers_b, r, q) in enumerate(zip(pyramid_a, pyramid_b, 
                                                       self.adj_rng, self.quant)):
            d = self.single_pyramid_matching(layers_a, layers_b, r, q, weight=weight)
            if norm:
                a = self.single_pyramid_matching(layers_a, layers_a, r, q, weight=weight)
                b = self.single_pyramid_matching(layers_b, layers_b, r, q, weight=weight)
                dist += d / tf.math.sqrt(a*b)
            else:
                dist += d
        
        return dist
    
    
    def dist_calc(self, X, Y, sparse=True, weight="max", norm=True):
        a = self.pyramid_encode(X, sparse=sparse)
        b = self.pyramid_encode(Y, sparse=sparse)
        return self.pmk_dist(a, b, weight=weight, norm=norm)