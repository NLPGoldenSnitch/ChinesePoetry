from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import json

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Read the data.
file_name = 'all_2.txt'
dictionary_name = 'dict.txt'
rdictionary_name = 'rdict.txt'
debug_name = 'debug.txt'
f = open(file_name, "r", encoding='utf-8',)
d = open(dictionary_name, "w", encoding='utf-8',)
rd = open(rdictionary_name, "w", encoding='utf-8',)

dbg = open(debug_name, "w",  encoding='utf-8',)
vocabulary = list()
line = ''
for f_line in f:
  line += f_line.replace('\n','$')
for char in line:
    if char != ' ':
      vocabulary.append(char)

dbg.write(line)

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 10000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
ends = dictionary['$']
vocabulary_size = len(dictionary)

json.dump(dictionary, d)
json.dump(reverse_dictionary,rd)
dbg.close()
rd.close()
d.close()

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  touch_ends = False
  batch_ends = False
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    if touch_ends:
      break
    buffer.append(data[data_index])
    if data[data_index] == ends:
      touch_ends = True
    data_index = (data_index + 1) % len(data)
  i = 0
  buff_len = len(buffer)
  while i < batch_size:
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    j=0
    while (j < num_skips) and (j < buff_len - 1) and (len(buffer) >= (skip_window+1)):
      if buffer[skip_window] == ends:
        batch_ends = True
      if i >= batch_size:
        break
      while target in targets_to_avoid:
        target = random.randint(0, buff_len - 1)
      targets_to_avoid.append(target)
      batch[i] = buffer[skip_window]
      labels[i, 0] = buffer[target]
      i += 1
      j += 1
    if batch_ends or (len(buffer) < (skip_window + 1)):
      touch_ends = False
      for _ in range(span):
        if touch_ends:
          break
        buffer.append(data[data_index])
        if data[data_index] == ends:
          touch_ends = True
        data_index = (data_index + 1) % len(data)
      batch_ends = False
    else:
      if not touch_ends:
        buffer.append(data[data_index])
        if data[data_index] == ends:
          touch_ends = True
        data_index = (data_index + 1) % len(data)
      else:
        buffer.popleft()
    buff_len = len(buffer)
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=16, num_skips=2, skip_window=1)
for i in range(16):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 512  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  with tf.device('/cpu:0'):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name = 'embeddings')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

saver = tf.train.Saver({'embeddings':embeddings})

with tf.Session(graph=graph) as session:
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

    if step % 100000 == 0:
      saver.save(session, 'word2vec_embeddings')