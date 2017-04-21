# -*- coding: utf-8 -*-
# file: model.py
# author: Li Jie
# ------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import os
#from tensorflow.models.rnn import rnn_cell, seq2seq
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from data import process_poems, generate_batch


tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')

# set this to 'main.py' relative path
tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/poems/'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('poem.txt'), 'file name of poems.')


tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')

tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS

# tf.ops.reset_default_graph()
sess = tf.InteractiveSession()

def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)
    poems_vector, word_to_int, vocabularies,out_vector = process_poems(FLAGS.file_path)
    x_batches, y_batches,length,outlength = generate_batch(FLAGS.batch_size, poems_vector, word_to_int,out_vector)

    input_data = [tf.placeholder(tf.int32, (None,)) for t in range(length)]
    labels = [tf.placeholder(tf.int32, (None,)) for t in range(length)]
    output_targets = [tf.placeholder(tf.int32, (None,)) for t in range(outlength)]
    rnn_model("gru",input_data,output_targets,labels,len(vocabularies))
    for i in range(len(x_batches)):
        print("training epoch:"+str(i))
        xb = x_batches[i]
        yb = y_batches[i]
        for j in range(len(xb)):
            X = xb[j]
            Y = yb[j]
            feed_dict = {input_data[t]: X[t] for t in range(length)}
            feed_dict.update({labels[t]: Y[t] for t in range(length)})
            _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
        # loss_t,summary = train_batch(x_batches[i],y_batches[i])
            summary_writer.add_summary(summary,i)
    summary_writer.flush()


def rnn_model(model, input_data, output_data,labels, vocab_size, batch_size=64,rnn_size=128):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """

    end_points = {}

    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size)
    # cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

    outputs,last_state = seq2seq.embedding_rnn_seq2seq(input_data,output_data,cell,vocab_size,vocab_size,len(input_data))
    loss = seq2seq.sequence_loss(ou, labels, weights, vocab_size)

    tf.scalar_summary("loss", loss)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(last_state[1])))
    tf.scalar_summary("magnitude at t=1", magnitude)
    summary_op = tf.merge_all_summaries()

    learning_rate = 0.05
    momentum = 0.9  
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(loss)
    logdir = tempfile.mkdtemp()
    print(logdir)
    summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)
    # if output_data is not None:
    #     initial_state = cell.zero_state(batch_size, tf.float32)
    # else:
    #     initial_state = cell.zero_state(1, tf.float32)

    # with tf.device("/cpu:0"):
    #     embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
    #         [vocab_size + 1, rnn_size], -1.0, 1.0))
    #     inputs = tf.nn.embedding_lookup(embedding, input_data)
    #     decoder_inputs = tf.nn.embedding_lookup(embedding, output_data)

    # # [batch_size, ?, rnn_size] = [64, ?, 128]
    # # outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    # outputs,last_state = basic_rnn_seq2seq(inputs,decoder_inputs,cell)
    # output = tf.reshape(outputs, [-1, rnn_size])

    # weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    # bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    # logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # # [?, vocab_size+1]

    # if output_data is not None:
    #     # output_data must be one-hot encode
    #     labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
    #     # should be [?, vocab_size+1]

    #     loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #     # loss shape should be [?, vocab_size+1]
    #     total_loss = tf.reduce_mean(loss)
    #     train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    #     end_points['initial_state'] = initial_state
    #     end_points['output'] = output
    #     end_points['train_op'] = train_op
    #     end_points['total_loss'] = total_loss
    #     end_points['loss'] = loss
    #     end_points['last_state'] = last_state
    # else:
    #     prediction = tf.nn.softmax(logits)

    #     end_points['initial_state'] = initial_state
    #     end_points['last_state'] = last_state
    #     end_points['prediction'] = prediction

    # return end_points