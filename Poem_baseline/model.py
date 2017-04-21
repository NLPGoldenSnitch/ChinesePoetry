# -*- coding: utf-8 -*-
# file: model.py
# author: Li Jie
# ------------------------------------------------------------------------
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import rnn_cell, seq2seq
from data import DataHandle



class RNNModel(object):
    """
    The core recurrent neural network model.
    """
    def __init__(self,args,data,isGen,model="lstm"):
        """
            Args:
                args:
                data:
                ifGen: control the model if for training or genertaion
                       0: training
                       1: genertaion

        """
        if isGen == 1:
            args.batch_size = 1
            args.seq_length = 1

        with tf.name_scope('input'):
            self.input_data= tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.target_data= tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        with tf.name_scope('model'):
            self.cell = None
            if model =="lstm":
                self.cell = rnn_cell.BasicLSTMCell(args.cell_size)
            elif model =="gru":
                self.cell = rnn_cell.GRUCell(args.cell_size)
            elif model == 'rnn':
                self.cell = rnn_cell.BasicRNNCell(args.cell_size)
            self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)
            self.initial_state = self.cell.zero_state(
                args.batch_size, tf.float32)
            with tf.variable_scope('rnnlm'):
                w = tf.get_variable(
                    'softmax_w', [args.cell_size, data.vocab_size])
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [data.vocab_size, args.cell_size])
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            outputs, last_state = tf.nn.dynamic_rnn(
                self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(outputs,[-1,args.cell_size])

            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / args.batch_size
            tf.scalar_summary('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.scalar_summary('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.histogram_summary(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.merge_all_summaries()

