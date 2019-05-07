# -*- coding: utf-8 -*-
import tensorflow as tf


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]


def create_variable(name, shape):
    """
    Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.
    如果激活函数使用sigmoid和tanh,怎最好使用xavir
    :param name:
    :param shape:
    :return:
    """
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.get_variable(name=name, initializer=initializer(shape=shape), dtype=tf.float32)
    return variable


def LSTM(X, data_length, batch_size, n_hidden_units):
    """
    RNN预测
    :param X:
    :param data_length:
    :return:
    """
    # forget_bias -> 1:全部遗忘 0：完全记忆
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # (?,14,20) c,h=(?,20)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X, time_major=False, sequence_length=data_length,
                                             dtype=tf.float32)
    print(outputs)
    print(final_state)
    # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X, time_major=False,dtype=tf.float32)
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # final_state = tf.reshape(outputs[-1], [-1,n_hidden_units])   # states is the last outputs
    print("reshape final_state =", final_state)
    return outputs, final_state
