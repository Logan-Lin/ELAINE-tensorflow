import os
import pickle

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
import networkx as nx
import pandas as pd
import utils


# Load O-D set and K-Means
with open(os.path.join('data', 'chengdu', 'od_20161001.pkl'), 'rb') as od_file:
    od_set = pickle.load(od_file)
with open('model/kmeans.pkl', 'rb') as kmeans_file:
    kmeans = pickle.load(kmeans_file)


# Contruct graph
od_set_label = pd.DataFrame(od_set, copy=True)
od_set_label['cluster'] = kmeans.predict(od_set[['longitude', 'latitude']])
slot_graph = nx.DiGraph()
utils.add_timeslot_weight_to_graph(od_set_label, slot_graph)


# Implementation of random walk
from node2vec import Node2Vec
node2vec_class = Node2Vec(graph, dimensions=10, walk_length=8, num_walks=1600, weight_key='weight')
n2v = node2vec_class.fit(window=8, min_count=1, batch_words=8)


# Definition of graph data
NODE_NUM = len(graph.nodes())
NEIGHBOR_NUM = 4

# Edge attributes matrix
edge_attr = dict()
for start, end in slot_graph.edges():
    edge_attr[(start, end)] = slot_graph[start][end]['weight']
edge_attr_matrix = pd.DataFrame(edge_attr).T
edge_attr_matrix = edge_attr_matrix.sample(frac=1).astype(np.float32)
edge_attr_norm = pd.DataFrame(normalize(edge_attr_matrix))
edge_attr_norm.index = edge_attr_matrix.index

# Node attributes matrix
node_attr = dict()
for i in range(NODE_NUM):
    row = []
    for j in range(NODE_NUM):
        try:
            weight = graph[i][j]['weight']
        except KeyError:
            weight = 0
        row.append(weight)
    node_attr[i] = row
node_attr_matrix = pd.DataFrame(node_attr).T
node_attr_matrix = node_attr_matrix.sort_index().astype(np.float32)
node_attr_norm = pd.DataFrame(normalize(node_attr_matrix))
node_attr_norm.index = node_attr_matrix.index

# Neighborhood matrix
neig = []
for i in range(NODE_NUM):
    row = [node_attr_matrix[i]]
    for j in range(NEIGHBOR_NUM):
        near_row = n2v.wv.most_similar(str(i))[j]
        near_node = int(near_row[0])
        row.append(node_attr_matrix.loc[near_node])
    row_series = pd.concat(row)
    row_series.index = range(row_series.shape[0])
    neig.append(row_series)
neig_matrix = pd.DataFrame(neig)
neig_norm = pd.DataFrame(normalize(neig_matrix))
neig_norm.index = neig_matrix.index


# ELAINE model

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def encoder(input_x, weights, biases, fc_activation, out_mean, out_log_sigma):
    """
    VAE encoder. The length of List `weights` and `biases` indicates
    the depth of FC layer in this encoder.
    Maps inputs onto a normal distribution in latent space.
    The transformation is parameterized and can be learned.
    
    @params input_x: Input tensor.
    @params weights: List of FC layer weights.
    @params biases: List of FC layer biases.
    @params fc_activation: Activation function for FC layers.
    @params out_mean: Dict in format {'weight': out_mean_weight, 'bias': out_mean_bias}
    @params out_log_sigma: Dict in format {'weight': out_log_sigma_weight, 'bias': out_log_sigma_bias}
    """
    current_layer = input_x
    for weight, bias in zip(weights, biases):
        current_layer = fc_activation(tf.add(tf.matmul(current_layer, weight), bias))
    z_mean = tf.add(tf.matmul(current_layer, out_mean['weight']), out_mean['bias'])
    z_log_sigma_sq = tf.add(tf.matmul(current_layer, out_log_sigma['weight']), out_log_sigma['bias'])
    return (z_mean, z_log_sigma_sq)


def decoder(input_z, weights, biases, fc_activation, out_mean):
    """
    VAE decoder. The length of List `weights` and `biases` indicated
    the depth of the FC layer in this decoder.
    Maps points in latent space onto a Bernoulli distribution in data space.
    The transformation is parameterized and can be learned.
    
    @params input_z: Input tensor sampled from Gaussian distribution.
    @params weights: List of FC layer weights.
    @params biases: List of FC layer biases.
    @params fc_activation: Activation function for FC layers.
    @params out_mean: Dict in format {'weight': out_mean_weight, 'bias': out_mean_bias}
    """
    layers = []
    current_layer = input_z
    for weight, bias in zip(weights, biases):
        current_layer = fc_activation(tf.add(tf.matmul(current_layer, weight), bias))
        layers.append(current_layer)
    x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(current_layer, out_mean['weight']), out_mean['bias']))
    return x_reconstr_mean


# Encoder constants
ENCODER_INPUT_LENGTH = neig_matrix.shape[1]  # Single encoder input tensor length. Current: 2000
ENCODER_SIZES = [ENCODER_INPUT_LENGTH, 1024, 512, 256, 128]  # Sizes of each layer in encoder FC part.
LATENT_SIZE = 32  # Size of latent space, also the length of encoder output.
ENCODER_DEPTH = len(ENCODER_SIZES)

# Single decoder constants
SINGLE_DECODER_INPUT_LENGTH = LATENT_SIZE
SINGLE_DECODER_SIZES = [SINGLE_DECODER_INPUT_LENGTH, 128, 256, 512, 1024]  # Size of each layer in single decoder FC part.
SINGLE_DECODER_OUTPUT_LENGTH = ENCODER_INPUT_LENGTH
SINGLE_DECODER_DEPTH = len(SINGLE_DECODER_SIZES)

# Double decoder constants
DOUBLE_DECODER_INPUT_LENGTH = LATENT_SIZE * 2
DOUBLE_DECODER_SIZES = [DOUBLE_DECODER_INPUT_LENGTH, 64, 32]  # Sizes of each layer in double decoder FC part.
DOUBLE_DECODER_OUTPUT_LENGTH = edge_attr_matrix.shape[1]
DOUBLE_DECODER_DEPTH = len(DOUBLE_DECODER_SIZES)

# Basic model parameters
BATCH_SIZE = 50
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 100

NODE_ATTR_LOSS_WEIGHT = 1
LATENT_LOSS_WEIGHT = 0.01
EDGE_ATTR_LOSS_WEIGHT = 0.5


def next_batch(edge_attr, batch_size, steps):
    """
    Fetch next batch graph data.
    """
    edge_count = edge_attr.shape[0]
    
    start = steps*batch_size % edge_count
    end = (steps+1)*batch_size % edge_count
    
    if start > end:
        result = pd.concat([edge_attr.iloc[start:], edge_attr.iloc[:end]])
    else:
        result = edge_attr_matrix.iloc[start:end]
    return result


class ELAINE:
    """
    Implementation of ELAINE model.
    Paper: Goyal P, Hosseinmardi H, Ferrara E, et al. Capturing Edge Attributes via Network Embedding[J]. 
    arXiv preprint arXiv:1805.03280, 2018.
    """

    def __init__(self, batch_size, learning_rate):
        tf.reset_default_graph()
        # Get encoder and decoders' initial parameters.
        self._init_params()

        # Input placeholder for "left encoder" and "right encoder"
        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.l_node_input = tf.placeholder(dtype=tf.float32,
                                               shape=[None, ENCODER_INPUT_LENGTH], name='l_node')
            self.r_node_input = tf.placeholder(dtype=tf.float32,
                                               shape=[None, ENCODER_INPUT_LENGTH], name='r_node')

            self.edge_attr = tf.placeholder(dtype=tf.float32,
                                            shape=[None, DOUBLE_DECODER_OUTPUT_LENGTH], name='edge')

        # Assign learning rate and batch size.
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Build network structure.
        self._create_network()
        # Create loss tensor and optimize operation.
        self._create_loss_optimizer()

        # Run gloabal variables initializer.
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        """
        Build up network structure.
        """
        # Feed "left" and "right" node attribute matrixes into variational encoder.
        # Use encoder to determine mean and variance of Gaussian distribution in latent space.
        self.l_z_mean, self.l_z_log_sigma_sq = encoder(self.l_node_input,
                                                       self.e_p['weights'], self.e_p['biases'],
                                                       self.e_p['fc_activation'], self.e_p['out_mean'],
                                                       self.e_p['out_log_sigma'])
        self.r_z_mean, self.r_z_log_sigma_sq = encoder(self.r_node_input,
                                                       self.e_p['weights'], self.e_p['biases'],
                                                       self.e_p['fc_activation'], self.e_p['out_mean'],
                                                       self.e_p['out_log_sigma'])

        # Draw one sample form Gaussian distribution.
        l_eps = tf.random_normal(
            (self.batch_size, LATENT_SIZE), 0, 1, dtype=tf.float32)
        r_eps = tf.random_normal(
            (self.batch_size, LATENT_SIZE), 0, 1, dtype=tf.float32)
        # z = mu = sigma * epsilon
        l_z = tf.add(self.l_z_mean, tf.multiply(
            tf.sqrt(tf.exp(self.l_z_log_sigma_sq)), l_eps))
        r_z = tf.add(self.r_z_mean, tf.multiply(
            tf.sqrt(tf.exp(self.r_z_log_sigma_sq)), r_eps))

        # Use decoder to determine mean of Bernoulli distribution of reconstructed input.
        self.l_reconstr_mean = decoder(l_z,
                                       self.s_d_p['weights'], self.s_d_p['biases'],
                                       self.s_d_p['fc_activation'], self.s_d_p['out_mean'])
        self.r_reconstr_mean = decoder(r_z,
                                       self.s_d_p['weights'], self.s_d_p['biases'],
                                       self.s_d_p['fc_activation'], self.s_d_p['out_mean'])

        # Concatenate "left" and "right" encoder's result into matrix doubled the width.
        # The matrix is feeded into decoder and used to reconstruct edge attributes.
        edge_input = tf.concat([self.l_z_mean, self.r_z_mean], -1)
        self.edge_reconstr_mean = decoder(edge_input,
                                          self.d_d_p['weights'], self.d_d_p['biases'],
                                          self.d_d_p['fc_activation'], self.d_d_p['out_mean'])

    def _create_loss_optimizer(self):
        """
        Calculate nodes and labels' reconstruction losses and latent losses,
        then combine them into final loss.
        """
        l_reconstr_loss = -tf.reduce_sum(self.l_node_input * tf.log(1e-10 + self.l_reconstr_mean)
                                         + (1-self.l_node_input) *
                                         tf.log(1e-10 + 1 - self.l_reconstr_mean), 1)
        r_reconstr_loss = -tf.reduce_sum(self.r_node_input * tf.log(1e-10 + self.r_reconstr_mean)
                                         + (1-self.r_node_input) *
                                         tf.log(1e-10 + 1 - self.r_reconstr_mean), 1)

        l_latent_loss = -tf.reduce_sum(1 + self.l_z_log_sigma_sq
                                       - tf.square(self.l_z_mean)
                                       - tf.exp(self.l_z_log_sigma_sq), 1)
        r_latent_loss = -tf.reduce_sum(1 + self.r_z_log_sigma_sq
                                       - tf.square(self.r_z_mean)
                                       - tf.exp(self.r_z_log_sigma_sq), 1)

        edge_reconstr_loss = -tf.reduce_sum(self.edge_attr * tf.log(1e-10 + self.edge_reconstr_mean)
                                            + (1-self.edge_attr) *
                                            tf.log(1e-10 + 1 - self.edge_reconstr_mean), 1)

        self.cost = \
            NODE_ATTR_LOSS_WEIGHT * tf.reduce_sum(0.5 * l_reconstr_loss + 0.5 * r_reconstr_loss) + \
            LATENT_LOSS_WEIGHT * tf.reduce_sum(0.5 * l_latent_loss + 0.5 * r_latent_loss) + \
            EDGE_ATTR_LOSS_WEIGHT * tf.reduce_sum(edge_reconstr_loss)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, edge_batch):
        """
        Fit a batch of edge attribute matrix.
        """
        l_node_attrs, r_node_attrs, edge_attrs = [], [], []
        for node_pair, edge_attr in edge_batch.iterrows():
            l_node = node_pair[0]
            r_node = node_pair[1]

            l_node_attrs.append(neig_norm.loc[l_node])
            r_node_attrs.append(neig_norm.loc[r_node])

            edge_attrs.append(edge_attr)
        l_node_attrs = np.array(l_node_attrs)
        r_node_attrs = np.array(r_node_attrs)
        edge_attrs = np.array(edge_attrs)

        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.l_node_input: l_node_attrs,
                                             self.r_node_input: r_node_attrs,
                                             self.edge_attr: edge_attrs})
        return cost

    def transform(self, X):
        """
        Transform data by mapping it into the latent space.
        """
        return self.sess.run(self.l_z_mean, feed_dict={self.l_node_input: X})

    def _init_params(self):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            self.e_p = dict(
                weights=[tf.Variable(xavier_init(ENCODER_SIZES[i], ENCODER_SIZES[i+1]),
                                     name='weight_{}'.format(i))
                         for i in range(ENCODER_DEPTH-1)],
                biases=[tf.Variable(tf.zeros([ENCODER_SIZES[i+1]], dtype=tf.float32),
                                    name='bias_{}'.format(i))
                        for i in range(ENCODER_DEPTH-1)],
                fc_activation=tf.nn.softplus,
                out_mean=dict(
                    weight=tf.Variable(xavier_init(ENCODER_SIZES[-1], LATENT_SIZE),
                                       name='out_mean_weight'),
                    bias=tf.Variable(tf.zeros([LATENT_SIZE], dtype=tf.float32),
                                     name='out_mean_bias')
                ),
                out_log_sigma=dict(
                    weight=tf.Variable(xavier_init(ENCODER_SIZES[-1], LATENT_SIZE),
                                       name='out_log_sigma_weight'),
                    bias=tf.Variable(tf.zeros([LATENT_SIZE], dtype=tf.float32),
                                     name='out_mean_bias')
                )
            )
        with tf.variable_scope('s-decoder', reuse=tf.AUTO_REUSE):
            self.s_d_p = dict(
                weights=[tf.Variable(xavier_init(SINGLE_DECODER_SIZES[i], SINGLE_DECODER_SIZES[i+1]),
                                     name='weight_{}'.format(i))
                         for i in range(SINGLE_DECODER_DEPTH-1)],
                biases=[tf.Variable(tf.zeros([SINGLE_DECODER_SIZES[i+1]], dtype=tf.float32),
                                    name='bias_{}'.format(i))
                        for i in range(SINGLE_DECODER_DEPTH-1)],
                fc_activation=tf.nn.softplus,
                out_mean=dict(
                    weight=tf.Variable(xavier_init(SINGLE_DECODER_SIZES[-1], SINGLE_DECODER_OUTPUT_LENGTH),
                                       name='out_mean_weight'),
                    bias=tf.Variable(tf.zeros([SINGLE_DECODER_OUTPUT_LENGTH], dtype=tf.float32),
                                     name='out_mean_bias')
                )
            )
        with tf.variable_scope('d-decoder', reuse=tf.AUTO_REUSE):
            self.d_d_p = dict(
                weights=[tf.Variable(xavier_init(DOUBLE_DECODER_SIZES[i], DOUBLE_DECODER_SIZES[i+1]),
                                     name='weight_{}'.format(i))
                         for i in range(DOUBLE_DECODER_DEPTH-1)],
                biases=[tf.Variable(tf.zeros([DOUBLE_DECODER_SIZES[i+1]]), name='bias_{}'.format(i))
                        for i in range(DOUBLE_DECODER_DEPTH-1)],
                fc_activation=tf.nn.softplus,
                out_mean=dict(
                    weight=tf.Variable(xavier_init(DOUBLE_DECODER_SIZES[-1], DOUBLE_DECODER_OUTPUT_LENGTH),
                                       name='out_mean_weight'),
                    bias=tf.Variable(
                        tf.zeros([DOUBLE_DECODER_OUTPUT_LENGTH], dtype=tf.float32), name='out_mean_bias')
                )
            )


def train(batch_size, learning_rate, training_epochs, display_step=1):
    elaine = ELAINE(batch_size=batch_size, learning_rate=learning_rate)
#     writer = tf.summary.FileWriter('model/elaine/', elaine.sess.graph)
    
    batch_one_epoch = int(edge_attr_matrix.shape[0] / batch_size) + 1
    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(batch_one_epoch):
            edge_batch = next_batch(edge_attr_norm, batch_size, (epoch + 1) * i)
            cost = elaine.partial_fit(edge_batch)
            avg_cost += cost / edge_attr_matrix.shape[0] * batch_size
            if i % 100 == 0:
                print('    Batch: %04d, cost: %8.5f' % (i+1, cost))
        if epoch % display_step == 0:
            print('>===Epoch: %04d, cost: %8.5f===<' % (epoch+1, avg_cost))
    return elaine


elaine = train(50, 0.0001, 3)
print(elaine.transform(neig_norm.iloc[0:2]))