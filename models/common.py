import math
import tensorflow as tf

def random_weight(dim_in, dim_out, name=None, stddev=1.0):
    """generate weight randomly"""
    return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)


def random_bias(dim, name=None):
    """generate weight randomly"""
    return tf.Variable(tf.truncated_normal([dim]), name=name)


def random_scalar( name=None):
    return tf.Variable(0.0, name=name)


def DropoutWrappedLSTMCell(hidden_size, in_keep_prob, name=None):
    """generate LSTM cell with dropout"""
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
    return cell


def mat_weight_mul(mat, weight):
    # [batch_size, n, m] * [m, p] = [batch_size, n, p]
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    assert (mat_shape[-1] == weight_shape[0])
    mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
    mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
    return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])