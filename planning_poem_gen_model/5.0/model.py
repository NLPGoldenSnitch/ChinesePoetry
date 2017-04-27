# -*- coding: utf-8 -*-
# file: model.py
# author: Li Jie
# ------------------------------------------------------------------------
import os
import sys
import time
import copy

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from data import DataHandle



class RNNModel(object):
    """
    The core recurrent neural network model.
    """
    def __init__(self,args,data,isGen,model='gru'):
        """
            Args:
                args:
                data:
                ifGen: control the model if for training or genertaion
                       0: training
                       1: genertaion

        """
        if isGen == 1:
            self.batch_size = 1
        else:
            self.batch_size = args.batch_size


        s_num = args.poem_form[0]-1
        s_len = args.poem_form[1]+1

        #define input layer
        self.poem_data = []
        self.keyword_data = []
        self.decoder_data = []
        self.target_data = []
        for i in range(s_num*s_len):
            self.poem_data.append(tf.placeholder(tf.int32, [self.batch_size]))
        for i in range(s_len):
            self.target_data.append(tf.placeholder(tf.int32, [self.batch_size]))
            self.decoder_data.append(tf.placeholder(tf.int32, [self.batch_size]))
        for i in range(args.keyword_length):
            self.keyword_data.append(tf.placeholder(tf.int32, [self.batch_size]))

        #1.1.word2vec
        self.embeddings = tf.Variable(tf.zeros([data.getDictsize(), args.embedding_size]),
            trainable=False)
        embedded_poem = [tf.nn.embedding_lookup(self.embeddings, i) for i in self.poem_data]
        embedded_key = [tf.nn.embedding_lookup(self.embeddings, i) for i in self.keyword_data]
        embedded_decoder_in = [tf.nn.embedding_lookup(self.embeddings, i) for i in self.decoder_data]

        #1.2.bidirectional
        self.cell_encoder_fw = rnn.GRUCell(args.cell_size)
        self.cell_encoder_bw = rnn.GRUCell(args.cell_size)
        with tf.variable_scope('encoder_poem'):
            encoder_outputs, *encoder_poem_fstates = \
            rnn.static_bidirectional_rnn(self.cell_encoder_fw,
                                         self.cell_encoder_bw,
                                         embedded_poem,
                                         dtype = tf.float32
                                       )
        with tf.variable_scope('encoder_key'):
            _, *encoder_key_states = \
            rnn.static_bidirectional_rnn(self.cell_encoder_fw,
                                         self.cell_encoder_bw,
                                         embedded_key,
                                         dtype = tf.float32
                                       )
        encoder_key_output = tf.concat(encoder_key_states,axis=1)
        #encoder_outputs.insert(0,encoder_key_output)


        #2.attention mechanism + decoder
        self.cell_decoder = rnn.GRUCell(args.cell_size)
        self.attention_v = tf.Variable(tf.random_uniform([args.cell_size,1],-0.08,0.08))
        self.attention_w = tf.Variable(tf.random_uniform([args.cell_size,args.cell_size],-0.08,0.08))
        self.attention_u = tf.Variable(tf.random_uniform([2*args.cell_size,args.cell_size],-0.08,0.08))
        self.decoder_win = tf.Variable(tf.random_uniform([2*args.cell_size,args.cell_size],-0.08,0.08))
        self.decoder_watt = tf.Variable(tf.random_uniform([2*args.cell_size,args.cell_size],-0.08,0.08))

        initial_state = self.cell_decoder.zero_state(self.batch_size,tf.float32)
        state = initial_state
        decoder_outputs = []
        self.generate_outputid = []
        embedded_output_id = tf.nn.embedding_lookup(self.embeddings, self.decoder_data[0])

        #variables for projection part of loss computation but used in decoder when generating
        self.proj_w = tf.Variable(tf.random_uniform([args.cell_size,data.getDictsize()],-0.08,0.08))
        self.proj_b = tf.Variable(tf.random_uniform([data.getDictsize()],-0.08,0.08))
        ones_weight = [tf.ones_like(self.target_data,dtype=tf.float32) for i in range(s_len)]
        logits = []
        self.probs = []

        with tf.variable_scope('decoder_variable') as decoder_scope:
            for i in range(s_len):
                decoder_last_state = state
                #2.1.attention mechanism
                #attention_weight_j = 
                #softmax(attention_v*tanh(attention_w*decoder_last_state+attention_u*encoder_outputs))[j]
                score = []
                for j in range(s_len*s_num):
                    score_state = tf.matmul(decoder_last_state,self.attention_w) #[batch_size cell_size]
                    score_encoder = tf.matmul(encoder_outputs[j],self.attention_u) #[batch_size cell_size]
                    score_tanh = tf.tanh(score_state+score_encoder) #[batch_size cell_size]
                    #score_tanh = tf.tanh(score_state) #[batch_size cell_size]
                    score_e = tf.matmul(score_tanh,self.attention_v) #[batch_size 1]
                    score.append(tf.exp(score_e))
                score_sum = tf.add_n(score)
                attention_inputs = []
                for j in range(s_len*s_num):
                    weight = tf.div(score[j],score_sum)
                    attention_inputs.append(tf.multiply(weight,encoder_outputs[j]))
                attention_input = tf.add_n(attention_inputs)
                state_input = decoder_last_state + \
                    tf.matmul(attention_input,self.decoder_watt)
                if isGen == 1:
                    decoder_input = embedded_output_id + \
                    tf.matmul(encoder_key_output,self.decoder_win)
                else:
                    decoder_input = embedded_decoder_in[i] + \
                    tf.matmul(encoder_key_output,self.decoder_win)
                #2.2.decoder
                if i > 0:
                    decoder_scope.reuse_variables()
                decoder_output, state=self.cell_decoder(decoder_input,state_input)
                decoder_outputs.append(decoder_output)
                logits.append(tf.matmul(decoder_outputs[i],self.proj_w)+self.proj_b)
                self.probs.append(tf.nn.softmax(logits[i]))
                self.generate_outputid.append(tf.argmax(self.probs[i],axis=1))
                embedded_output_id = tf.nn.embedding_lookup(self.embeddings, self.generate_outputid[i])

        if isGen == 0:
            #3.loss
            #3.2.loss
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(logits,
                                                        self.target_data,
                                                        ones_weight)

            #4.optimizer
            params = tf.trainable_variables()
            self.lr = tf.placeholder(tf.float32, [])
            opt = tf.train.GradientDescentOptimizer(self.lr)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,args.max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, params))