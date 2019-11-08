#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.framework import get_variables
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops

import imageio
import platform
import ot

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



class Trainer(object):
    
    def __init__(self, opts): #, feature_dims, net):
        # Load data
        x_s, y_s, x_t = load_data()
        
        batch_size = y_s.shape[0]
        self.gamma = tf.Variable(tf.zeros([batch_size, batch_size], dtype=tf.float32), name="gamma")
        self.opts = opts
        

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
        
    def rs1_cost(self, p1, p2, h, f):
        v1 = tf.random.sample()
        #g1 =  (f(p1, p2 + h * v1) - f(p1, p2 - h * v1) * v1) / 2h
        #g2 =  (f(p1 + h * v2, p2) - f(p1 - h * v2, p2) * v2) / 2h
        #g = g1 + g2 / 2
                
    
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
                        choices=["source_only", "adapt_mcd", "adapt_swd", "full_wass", "sinkhorn"])
    
    
    opts = parser.parse_args()
    trainer = Trainer(opts)
    trainer.train()
    
    
