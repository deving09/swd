#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.framework import get_variables
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops

import imageio
import platform
import ot
import pmk

from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('pdf')
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('pdf')
    #matplotlib.use('Agg')
    #matplotlib.use('TkAgg')





def toyNet(X):
    # Define network architecture
    with tf.variable_scope('Generator'):
        net = fully_connected(X, 15, activation_fn=tf.nn.relu)
        net = fully_connected(net, 15, activation_fn=tf.nn.relu)
        net = fully_connected(net, 15, activation_fn=tf.nn.relu)
    with tf.variable_scope('Classifier1'):
        net1 = fully_connected(net, 15, activation_fn=tf.nn.relu)
        net1 = fully_connected(net1, 15, activation_fn=tf.nn.relu)
        net1 = fully_connected(net1, 1, activation_fn=None)
        logits1 = tf.sigmoid(net1)
    with tf.variable_scope('Classifier2'):
        net2 = fully_connected(net, 15, activation_fn=tf.nn.relu)
        net2 = fully_connected(net2, 15, activation_fn=tf.nn.relu)
        net2 = fully_connected(net2, 1, activation_fn=None)
        logits2 = tf.sigmoid(net2)
    return logits1, logits2


def load_data():
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    moon_data = np.load('moon_data.npz')
    x_s = moon_data['x_s']
    y_s = moon_data['y_s']
    x_t = moon_data['x_t']
    return x_s, y_s, x_t



    
def sort_rows(matrix, num_rows):
    matrix_T = array_ops.transpose(matrix, [1, 0])
    sorted_matrix_T = nn_ops.top_k(matrix_T, num_rows)[0]
    return array_ops.transpose(sorted_matrix_T, [1, 0])
    
    
### Deconstructed PMK
@tf.custom_gradient
def rs2_cost(p1, p2):
    
    print(p1.shape)

    # PMK key variables
    min_n = -5.0
    rng = 10.0
    quants = [32, 24, 18]
    shifts = [np.random.random() * rng/q for q in quants]
    true_shifts = [ rng/q * s   for q, s in zip(quants, shifts)]
    adj_rngs = [rng + rng/q * s   for q, s in zip(quants, shifts)]
    
    # Custom gradient key variables
    h = 0.1
    v1 = tf.random.normal(tf.shape(p2)) * h
    v2 = tf.random.normal(tf.shape(p1)) * h
    norm_v1 = tf.norm(v1)
    norm_v2 = tf.norm(v2)
    
    def _encode(X, quant, shift, rng):
        layers = []
        while quant >= 1:
            bin_locations = tf.cast(tf.math.floor((X + shift - min_n) * quant/rng),tf.int64)
            dense_shape = tf.convert_to_tensor([quant] * X.shape[1], dtype=tf.int64) 
            values = tf.ones(tf.shape(X)[0], dtype=tf.float32)
            sparse_vec = tf.sparse.SparseTensor(indices=bin_locations, 
                                      values=values, dense_shape=dense_shape)
            layers.append(sparse_vec)
            quant = math.floor(quant/2)
        
        return layers 
    
    def pyramid_encode(X):
        pyramid = []
        for i, (q, s, r) in enumerate(zip(quants, true_shifts, adj_rngs)):
            pyramid.append(_encode(X, q, s, r))
        
        return pyramid
    
    def single_pyramid_matching(layers_1, layers_2, rng, quant, weight="max"):
        scores = 0.0
        matches = [0.0]
        for i, (l1, l2) in enumerate(zip(layers_1, layers_2)):
            if weight == 'max':
                layer_weight = math.sqrt(math.pow(rng/quant, 2) * len(l1.shape))
            elif weight == "min":
                layer_weight = math.sqrt(math.pow(rng/(quant*4.0), 2))
            elif weight == "mean":
                layer_weight = math.sqrt(math.pow(rng/(quant*2.0), 2))
            elif weight == "flat":
                layer_weight = rng/quant
                
            
            #mt = tf.stack([l1, l2])
            match_count = tf.sparse.reduce_sum(tf.sparse.minimum(l1, l2))
            matches.append(match_count)
            scores += layer_weight * (match_count - matches[i])
            quant = math.floor(quant/2)
        
        return scores
    
    def pmk_dist(pyramid_a, pyramid_b, weight="max", norm=True):
        #dist = tf.Variable(0.0, name="dist")
        dist = 0.0
        for i, (layers_a, layers_b, r, q) in enumerate(zip(pyramid_a, pyramid_b, 
                                                       adj_rngs, quants)):
            d = single_pyramid_matching(layers_a, layers_b, r, q, weight=weight)
            if norm:
                a = single_pyramid_matching(layers_a, layers_a, r, q, weight=weight)
                b = single_pyramid_matching(layers_b, layers_b, r, q, weight=weight)
                dist += d / tf.math.sqrt(a*b)
            else:
                dist += d
        
        return dist
    
    def dist_calc(X, Y, sparse=True, weight="max", norm=True):
        a = pyramid_encode(X)
        b = pyramid_encode(Y)
        return pmk_dist(a, b, weight=weight, norm=norm)
    
    
    def grad(dy):
        g1 =  (dist_calc(p1, p2 + v1) - dist_calc(p1, p2 - v1)) * v1 / (2 * norm_v1)
        g2 =  (dist_calc(p1 + v2, p2) - dist_calc(p1 + v2, p2)) * v2 / (2 * norm_v2)
        g = g1 + g2 / 2
        #print(g.shape)
        return g1 , g2
    
    return dist_calc(p1, p2), grad



class Trainer(object):
    
    def __init__(self, opts): #, feature_dims, net):
        # Load data
        x_s, y_s, x_t = load_data()
        
        batch_size = y_s.shape[0]
        self.gamma = tf.Variable(tf.zeros([batch_size, batch_size], dtype=tf.float32), name="gamma")
        self.opts = opts
        if self.opts.mode == "pmk":
            #REPLACE Hardcoded to test
            self.pmker = pmk.PMK(10, min_n = -5.0)
        
        self.h = opts.alpha
        
        

    def discrepancy_slice_wasserstein(self, p1, p2):
        s = array_ops.shape(p1)
        if p1.get_shape().as_list()[1] > 1:
            # For data more than one-dimensional, perform multiple random projection to 1-D
            proj = random_ops.random_normal([array_ops.shape(p1)[1], 128])
            proj *= math_ops.rsqrt(math_ops.reduce_sum(math_ops.square(proj), 0, keep_dims=True))
            p1 = math_ops.matmul(p1, proj)
            p2 = math_ops.matmul(p2, proj)
        p1 = sort_rows(p1, s[0])
        p2 = sort_rows(p2, s[0])
        wdist = math_ops.reduce_mean(math_ops.square(p1 - p2))
        return math_ops.reduce_mean(wdist)
    
    def discrepancy_mcd(self, out1, out2):
        return tf.reduce_mean(tf.abs(out1 - out2))
    
    
    def ot_matching_full(self, p1, p2, trans="emd"):
        c0 = cdist(p1, p2, metric='sqeuclidean')
        if trans == "sinkhorn":
            gamma = ot.emd(ot.unif(p1.shape[0]), ot.unif(p2.shape[0]), c0)
        else:
            gamma = ot.sinkhorn(ot.unif(p1.shape[0]), ot.unif(p2.shape[0]), c0, 1e-1)
        self.gamma = tf.assign(self.gamma, gamma)
    
    def discrepancy_full_wasserstein(self, p1, p2):
        p1_d  = tf.reshape(math_ops.reduce_sum(math_ops.square(p1), 1), (-1,1))
        p2_d  = tf.reshape(math_ops.reduce_sum(math_ops.square(p2), 1), (1,-1))
        p_cross = math_ops.matmul(p1, array_ops.transpose(p2))
        cost =p1_d + p2_d - 2.0 * p_cross
        return math_ops.reduce_sum(self.gamma * cost)
    
    @tf.custom_gradient
    def rs1_cost(self, p1, p2):
        v1 = tf.random.normal(p2.shape)
        v2 = tf.random.sample(p1.shape)
        def grad(dy):
            g1 =  (self.pmker.dist_calc(p1, p2 + self.h * v1) - self.pmker.dist_calc(p1, p2 - self.h * v1)) * v1 / (2 * self.h)
            g2 =  (self.pmker.dist_calc(p1 + self.h * v2, p2) - self.pmker.dist_calc(p1 + self.h * v2, p2)) * v2 / (2 * self.h)
            g = g1 + g2 / 2
        
        return self.pmker.dist_calc(p1, p2), grad                
    
    def train(self):

        # Load data
        x_s, y_s, x_t = load_data()
        
        def generate_grid_point():
            x_min, x_max = x_s[:, 0].min() - .5, x_s[:, 0].max() + 0.5
            y_min, y_max = x_s[:, 1].min() - .5, x_s[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
            return xx, yy

        # set random seed
        tf.set_random_seed(1234)

        # Define TF placeholders
        X = tf.placeholder(tf.float32, shape=[None, 2])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        X_target = tf.placeholder(tf.float32, shape=[None, 2])

        # Network definition
        with tf.variable_scope('toyNet'):
            logits1, logits2 = toyNet(X)
        with tf.variable_scope('toyNet', reuse=True):
            logits1_target, logits2_target = toyNet(X_target)

        # Cost functions
        eps = 1e-05
        cost1 = -tf.reduce_mean(Y * tf.log(logits1 + eps) + (1 - Y) * tf.log(1 - logits1 + eps))
        cost2 = -tf.reduce_mean(Y * tf.log(logits2 + eps) + (1 - Y) * tf.log(1 - logits2 + eps))
        loss_s = cost1 + cost2

        if self.opts.mode == 'adapt_swd':
            loss_dis = self.discrepancy_slice_wasserstein(logits1_target, logits2_target)
        elif self.opts.mode == 'full_wass' or self.opts.mode == 'sinkhorn' :
            loss_dis = self.discrepancy_full_wasserstein(logits1_target, logits2_target)
        elif self.opts.mode == "pmk":
            #loss_dis = self.pmker.dist_calc(logits1_target, logits2_target)
            #loss_dis = self.rs1_cost(logits1_target, logits2_target)
            loss_dis = rs2_cost(logits1_target, logits2_target)
        else:
            loss_dis = self.discrepancy_mcd(logits1_target, logits2_target)

        # Setup optimizers
        variables_all = get_variables(scope='toyNet')
        variables_generator = get_variables(scope='toyNet' + '/Generator')
        variables_classifier1 = get_variables(scope='toyNet' + '/Classifier1')
        variables_classifier2 = get_variables(scope='toyNet' + '/Classifier2')

        optim_s = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
            minimize(loss_s, var_list=variables_all)
        optim_dis1 = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
            minimize(loss_s - loss_dis, var_list=variables_classifier1)
        optim_dis2 = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
            minimize(loss_s - loss_dis, var_list=variables_classifier2)
        optim_dis3 = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
            minimize(loss_dis, var_list=variables_generator)

        # Select predictions from C1
        predicted1 = tf.cast(logits1 > 0.5, dtype=tf.float32)

        # Generate grid points for visualization
        xx, yy = generate_grid_point()

        # For creating GIF purpose
        gif_images = []

        # Start session
        with tf.Session() as sess:
            if opts.mode == 'source_only':
                print('-> Perform source only training. No adaptation.')
                train = optim_s
            else:
                print('-> Perform training with domain adaptation.')
                train = tf.group(optim_s, optim_dis1, optim_dis2, optim_dis3)

            # Initialize variables
            net_variables = tf.global_variables() + tf.local_variables()
            sess.run(tf.variables_initializer(net_variables))

            # Training
            for step in range(10001):
                if step % 1000 == 0:
                    print("Iteration: %d / %d" % (step, 10000))
                    Z = sess.run(predicted1, feed_dict={X: np.c_[xx.ravel(), yy.ravel()]})
                    Z = Z.reshape(xx.shape)
                    f = plt.figure()
                    plt.contourf(xx, yy, Z, cmap=plt.cm.copper_r, alpha=0.9)
                    plt.scatter(x_s[:, 0], x_s[:, 1], c=y_s.reshape((len(x_s))),
                                cmap=plt.cm.coolwarm, alpha=0.8)
                    plt.scatter(x_t[:, 0], x_t[:, 1], color='green', alpha=0.7)
                    plt.text(1.6, -0.9, 'Iter: ' + str(step), fontsize=14, color='#FFD700',
                             bbox=dict(facecolor='dimgray', alpha=0.7))
                    plt.axis('off')
                    f.savefig(opts.mode + '_iter' + str(step) + ".png", bbox_inches='tight',
                              pad_inches=0, dpi=100, transparent=True)
                    gif_images.append(imageio.imread(
                                      opts.mode + '_iter' + str(step) + ".png"))
                    plt.close()
                
                if self.opts.mode == "full_wass":
                    p1, p2 = sess.run([logits1, logits2], 
                                                  feed_dict={X: x_s, Y: y_s, X_target:x_t})
                    self.ot_matching_full(p1, p2)
                if self.opts.mode == "sinkhorn":
                    p1, p2 = sess.run([logits1, logits2], 
                                                  feed_dict={X: x_s, Y: y_s, X_target:x_t})
                    self.ot_matching_full(p1, p2, trans="sinkhorn")  
                    
                # Forward and backward propagation
                _ = sess.run([train], feed_dict={X: x_s, Y: y_s, X_target: x_t})

            # Save GIF
            imageio.mimsave(opts.mode + '.gif', gif_images, duration=0.8)
            print("[Finished]\n-> Please see the current folder for outputs.")
        return
    
    def other(self):
        return





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="adapt_swd",
                        choices=["source_only", "adapt_mcd", 
                        "adapt_swd", "full_wass", "sinkhorn", "pmk"])
    parser.add_argument( '--alpha', type=float, default=0.1)
    
    
    opts = parser.parse_args()
    trainer = Trainer(opts)
    trainer.train()
    
    
