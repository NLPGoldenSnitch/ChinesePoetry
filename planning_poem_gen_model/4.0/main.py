# python
# -*- coding: utf-8 -*-
# file: data.py
# author: Hongyi Ren
# ------------------------------------------------------------------------
import collections
import os
import sys
import numpy as np
import tensorflow as tf

from data import DataHandle
from model import RNNModel


class ControlParm(object):
    batch_size = 32
    n_epoch = 1000
    learning_rate = 0.5
    decay_steps = 100
    decay_rate = 0.99
    embedding_size = 128
    max_gradient_norm = 5.0

    recover_iter = 320000

    cell_size = 128
    keyword_length = 5
    poem_form = [4,7] #4x7.txt

def run(isGen):
    file_poem = "data/poem4x7.txt"
    file_keyword = "data/keyword4x7.txt"
    args = ControlParm()
    data = DataHandle(file_poem,file_keyword, args)
    model = RNNModel(args, data, isGen=isGen)

    if isGen == 0:
    #training mode
        print("------[INFO] Start trainning------")
        with tf.Session() as sess:
            #batch_times = data.size//args.batch_size
            batch_times = args.n_epoch * \
            data.size // args.batch_size
            print("[INFO] Total batch :" + str(batch_times))
            embedding_saver = tf.train.Saver({'embeddings':model.embeddings})
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            embedding_saver.restore(sess, './data/w2v_128/word2vec_embeddings')
            if args.recover_iter != -1:
                saver.restore(sess,'./model/poem_model.ckpt-'+str(args.recover_iter))
            for i in range(args.recover_iter+1,batch_times):
                learning_rate = args.learning_rate * (args.decay_rate ** 
                    (i // args.decay_steps))
                if learning_rate <= 0.0000000001:
                    learning_rate = 0.0000000001
                x_batch,y_batch,k_batch = data.generateBatch()
                feed_dict = {}
                for a in range((args.poem_form[1]+1)*(args.poem_form[0]-1)):
                    feed_dict[model.poem_data[a]] = x_batch[a]
                for a in range(args.poem_form[1]+1):
                    if a == 0:
                        feed_dict[model.decoder_data[a]] = [0 for b in range(args.batch_size)]
                    feed_dict[model.decoder_data[a]] = y_batch[a-1]
                    feed_dict[model.target_data[a]] = y_batch[a]
                for a in range(args.keyword_length):
                    feed_dict[model.keyword_data[a]] = k_batch[a]
                feed_dict[model.lr] = learning_rate
                train_loss,probs,_ = sess.run([model.loss,model.probs,model.update],feed_dict)
                if i%10 == 0:
                    print('[TRAINING] Step:{}/{}, training_loss:{}'.format(i,batch_times, train_loss))
                if i%2000 == 0 or (i+1) == batch_times:
                    print('[TRAINING] Step:{}/{}, training_loss:{}'.format(i,batch_times, train_loss))
                    print('[TRAINING] LR:', learning_rate)
                    for j in range(10):
                        setence = ''
                        true = ''
                        for k in range(args.poem_form[1]+1):
                            setence += data.convertId2Word(np.argmax(probs[k][j]))
                            true += data.convertId2Word(y_batch[k][j])
                        print(setence,'-',true)
                    saver.save(sess,'./model/poem_model.ckpt', global_step=i)
            print("------[INFO] Trainning finish------")

    elif isGen == 1:
    #generate mode
        print("------[INFO] Start generating poem------")
        with tf.Session() as sess:
            embedding_saver = tf.train.Saver({'embeddings':model.embeddings})
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            embedding_saver.restore(sess, './data/w2v_128/word2vec_embeddings')
            if args.recover_iter != -1:
                saver.restore(sess,'./model/poem_model.ckpt-'+str(args.recover_iter))
            sentence = [0 for i in range((args.poem_form[1]+1)*(args.poem_form[0]-1))]
            key = ['儒家','君','剑','金']
            keyword = [data.convertKey2Id(key[0]),data.convertKey2Id(key[1]),
                data.convertKey2Id(key[2]),data.convertKey2Id(key[3])]
            for i in range(args.poem_form[0]):
                feed_dict = {}
                for j in range((args.poem_form[1]+1)*(args.poem_form[0]-1)):
                    feed_dict[model.poem_data[j]] = [sentence[j]]
                for j in range(args.keyword_length):
                    feed_dict[model.keyword_data[j]] = [keyword[i][j]]
                for j in range(args.poem_form[1]+1):
                    feed_dict[model.target_data[j]] = [0]
                    feed_dict[model.decoder_data[j]] = [2]
                new_sentence = sess.run(model.generate_outputid,feed_dict)
                result = ''
                for j in new_sentence:
                    result+=data.convertId2Word(j[0])
                print(key[i],':',result)
                if i <3:
                    for j in range(len(new_sentence)):
                        sentence[i*(args.poem_form[1]+1)+j] = new_sentence[j][0]
    else:
    #for debug
        pass

if __name__ =="__main__":
    msg = """
    Usage:
    Training: 
        python3 main.py 0
    Generating:
        python3 main.py 1
    """
    if len(sys.argv) == 2:
        isGen = int(sys.argv[-1])
        print('--Sampling--' if isGen else '--Training--')
        run(isGen)
    else:
        print(msg)
        sys.exit(1)
