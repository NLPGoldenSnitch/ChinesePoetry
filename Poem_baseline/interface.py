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
from model import RNNModel
import pdb



class Interface(object):
    def __init__(self):
        pass

    def trainSeq(self,data,model,args):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            writer = tf.train.SummaryWriter(args.log_dir, sess.graph)           

            # Add embedding tensorboard visualization. Need tensorflow version
            # >= 0.12.0RC0
            config = projector.ProjectorConfig()
            embed = config.embeddings.add()
            embed.tensor_name = 'rnnlm/embedding:0'
            embed.metadata_path = args.metadata
            projector.visualize_embeddings(writer, config)
            state = sess.run(self.cell.zero_state(1, tf.float32))



    def train(self,data, model, args):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            writer = tf.train.SummaryWriter(args.log_dir, sess.graph)           

            # Add embedding tensorboard visualization. Need tensorflow version
            # >= 0.12.0RC0
            config = projector.ProjectorConfig()
            embed = config.embeddings.add()
            embed.tensor_name = 'rnnlm/embedding:0'
            embed.metadata_path = args.metadata
            projector.visualize_embeddings(writer, config)

            #batch_times = data.size//args.batch_size
            batch_times = args.n_epoch * \
            (data.size // args.seq_length) // args.batch_size
            #pdb.set_trace()
            print("[INFO] total batch :" + str(batch_times))
            for i in range(batch_times):
                #print("[INFO] Start Training Batch # : "+str(i+1))
                learning_rate = args.learning_rate * (args.decay_rate ** (i // args.decay_steps))
                x_batch,y_batch = data.generate_batch()

                #pdb.set_trace()
                feed_dict = {model.input_data: x_batch,
                             model.target_data: y_batch, model.lr: learning_rate}
                #pdb.set_trace()
                train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                     feed_dict)
                if i % 10 == 0:
                    writer.add_summary(summary, global_step=i)
                    print('Step:{}/{}, training_loss:{:4f}'.format(i,
                                                               batch_times, train_loss))
                if i % 2000 == 0 or (i + 1) == batch_times:
                    saver.save(sess, os.path.join(args.log_dir, 'poem_model.ckpt'), global_step=i)       

        print("[INFO] trainning finish")    


    def generate(self,data, model, args, topic):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(args.log_dir)
            print(ckpt)
            saver.restore(sess, ckpt)
            prime = topic
            state = sess.run(model.cell.zero_state(1, tf.float32))

            for word in prime[:-1]:
                x = np.zeros((1, 1))
                x[0, 0] = data.convertWord2Id(word)
                feed = {model.input_data: x, model.initial_state: state}
                state = sess.run(model.last_state, feed)

            word = prime[-1]
            poem = prime
            for i in range(args.gen_num):
                x = np.zeros([1,1])
                x[0,0] = data.convertWord2Id(word)
                feed_dict = {model.input_data: x, model.initial_state: state}
                probs, state = sess.run([model.probs, model.last_state], feed_dict)
                p = probs[0]
                word = data.convertId2Word(np.argmax(p)) 
                print(word, end='')
                sys.stdout.flush()
                time.sleep(0.05)
                poem += word    
            return poem   