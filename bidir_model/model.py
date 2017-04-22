# -*- coding: utf-8 -*-
# file: model.py
# author: Li Jie
# ------------------------------------------------------------------------
import os
import sys
import time
import pdb
import numpy as np
import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.python.ops import rnn_cell
from data import DataHandleForSeqence
from bidir_attn_seq2seq import bidir_attn_seq2seq



class RNNModel(object):
    """
    The core recurrent neural network model.
    """
    def __init__(self,args,data,isGen,model="gru"):
        """
            Args:
                args:
                data:
                ifGen: control the model if for training or genertaion
                       0: training
                       1: genertaion

        """
        feed = False
        if isGen == 1:
            feed = True
            args.batch_size = 1


        with tf.name_scope('input'):
            self.input_data=  [tf.placeholder(tf.int32, [args.batch_size],name ="input{0}".format(i)) for i in range(args.seq_length)]
            self.target_data= [tf.placeholder(tf.int32, [args.batch_size],name ="target{0}".format(i)) for i in range(args.out_length)]
            self.decode_input= [tf.placeholder(tf.int32, [args.batch_size],name ="target{0}".format(i)) for i in range(args.out_length)]
            self.weight_data= [tf.placeholder(tf.float32, [args.batch_size],name ="weight{0}".format(i)) for i in range(args.out_length)]
            # self.target_data= [tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            # self.weight_data= [tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        with tf.name_scope('model'):
            self.cell = None
            if model =="lstm":
                # self.cell = rnn_cell.BasicLSTMCell(args.cell_size)
                self.cell = tf.contrib.rnn.BasicLSTMCell
            elif model =="gru":
                # self.cell = rnn_cell.GRUCell(args.cell_size)
                self.cell = tf.contrib.rnn.GRUCell
            elif model == 'rnn':
                self.cell = tf.nn.rnn_cell.BasicRNNCell(args.cell_size)
            # self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)
            # self.initial_state = self.cell.zero_state(
            #     args.batch_size, tf.float32)
            with tf.variable_scope('rnnlm'):
                w_t = tf.get_variable(
                    'softmax_w', [data.vocab_size,args.cell_size])
                w = tf.transpose(w_t)
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [data.vocab_size, args.cell_size])
                    inputs = [tf.nn.embedding_lookup(embedding, i) for i in self.input_data]
            # outputs, last_state = tf.nn.dynamic_rnn(
            #     self.cell, inputs, initial_state=self.initial_state)
            # self.targets = self.target_data[1:]+[[0]*args.batch_size]
            # self.decode_input = self.target_data
            # for l in range(args.batch_size):
            #     self.decode_input[6][l] = 0
            outputs, last_state = bidir_attn_seq2seq(
                inputs,self.decode_input,self.cell,args.cell_size,data.vocab_size,input_embedding=False,dtype = tf.float32, feed_previous = feed
                    #,output_projection=(w,b)
                    )

        with tf.name_scope('loss'):
            #output = tf.reshape(outputs,[-1,args.cell_size])

            #pdb.set_trace()
            #self.logits = tf.matmul(output, w) + b
            self.logits = outputs
            #pdb.set_trace()
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            #targets = tf.reshape(self.target_data, [-1])

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=args.batch_size,
                    num_classes=data.vocab_size),
                    dtype=tf.float32)
            
            loss = tf.contrib.legacy_seq2seq.sequence_loss(self.logits,
                                                    self.target_data,
                                                    self.weight_data,
                                                    #softmax_loss_function = sampled_loss
                                                    )
            #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(self.logits,
            #                                        targets,
            #                                        self.weight_data)
            self.cost = tf.reduce_sum(loss) / args.batch_size
            #self.cost = 2.79**tf.reduce_sum(loss)#
            #tf.contrib.deprecated.scalar_summary('loss', self.cost)
            tf.summary.scalar('loss', self.cost)


        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            #tf.contrib.deprecated.scalar_summary('learning_rate', self.lr)
            tf.summary.scalar('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            # for g in grads:
            #     tf.summary.histogram('histogram', g)
            # #     tf.contrib.deprecated.histogram_summary(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()

