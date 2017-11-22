import tensorflow as tf
import ipdb
import utils


CONFIG = {
    'dim_hidden': 40,
    'num_hidden_layers': 2,
}


def construct_weights(dim_input, dim_output):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal(
        [dim_input, CONFIG['dim_hidden']], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros([CONFIG['dim_hidden']]))
    for i in range(1, CONFIG['num_hidden_layers']):
        weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal(
            [CONFIG['dim_hidden'], CONFIG['dim_hidden']], stddev=0.01))
        weights['b'+str(i+1)] = tf.Variable(tf.zeros([CONFIG['dim_hidden']]))
    weights['w'+str(CONFIG['num_hidden_layers']+1)] = tf.Variable(
        tf.truncated_normal([CONFIG['dim_hidden'], dim_output], stddev=0.01))
    weights['b'+str(CONFIG['num_hidden_layers']+1)] = tf.Variable(
        tf.zeros([dim_output]))
    return weights


def construct_forward(inp, weights, reuse, norm, is_train):
    h = utils.contruct_layer(tf.matmul(inp, weights['w1']) + weights['b1'],
                             activation_fn=tf.nn.relu, reuse=reuse, is_train=is_train,
                             norm=norm, scope='1')
    for i in range(1, CONFIG['num_hidden_layers']):
        w = weights['w'+str(i+1)]
        b = weights['b'+str(i+1)]
        h = utils.contruct_layer(tf.matmul(h, w)+b, activation_fn=tf.nn.relu,
                                 reuse=reuse, norm=norm, is_train=is_train,
                                 scope=str(i+1))
    w = weights['w'+str(CONFIG['num_hidden_layers']+1)]
    b = weights['b'+str(CONFIG['num_hidden_layers']+1)]
    out = tf.matmul(h, w) + b
    return out
