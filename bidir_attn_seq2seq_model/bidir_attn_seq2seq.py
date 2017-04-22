import tensorflow as tf

def bidir_attn_seq2seq(
      enc_inputs,
      dec_inputs,
      cell,
      cell_size,
      enc_vocab_size,
      dec_vocab_size,
      key_inputs = None,
      key_length = None,
      num_heads = 1,
      input_embedding = True,
      output_projection = None,
      initial_state_fw = None,
      initial_state_bw = None,
      feed_previous = False,
      dtype = None,
      scope = None,
      initial_state_attention = False
    ):
  """
  Args:
    enc_inputs: a list of 2D int32 Tensor of shape [batch_size, cell_size]
    dec_inputs: a list of 2D int32 Tensor of shape [batch_size, cell_size]
    cell: a RNN cell function
    cell_size: dimension for input data
    enc_vocab_size: vocabulary size
  """
  with tf.variable_scope(scope or "bidir_attn_seq2seq", dtype=dtype) as scope:
    # keyword bi-directional RNN
    if key_inputs and key_length is not None:
      key_cell_fw = cell(cell_size)
      key_cell_bw = cell(cell_size)
      _, key_states = tf.nn.bidirectional_dynamic_rnn(
                        key_cell_fw,
                        key_cell_bw,
                        key_inputs,
                        sequence_length = key_length,
                        time_major = True
                      )
      initial_state_fw = key_states[0]
      initial_state_bw = key_states[1]

    # encoder
    if(input_embedding == True):
      embedding = tf.get_variable("embedding", [enc_vocab_size, cell_size])
      embedded_inputs = [tf.nn.embedding_lookup(embedding, batch) for batch in enc_inputs]
    else:
      embedded_inputs = enc_inputs
    enc_cell_fw = cell(cell_size)
    enc_cell_bw = cell(cell_size)
    enc_outputs, *enc_states = tf.contrib.rnn.static_bidirectional_rnn(
                                         enc_cell_fw,
                                         enc_cell_bw,
                                         embedded_inputs,
                                         initial_state_fw = initial_state_fw,
                                         initial_state_bw = initial_state_bw,
                                         dtype = dtype
                                       )
    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [
      tf.reshape(e, [-1, 1, cell_size * 2]) for e in enc_outputs
    ]
    attention_states = tf.concat(top_states, 1)

    enc_state = tf.concat(enc_states, 1)
    enc2dec_w = tf.get_variable("enc2dec_w", [cell_size * 2, cell_size])
    enc2dec_b = tf.get_variable("enc2dec_b", [cell_size])

    # decoder
    dec_cell = cell(cell_size)
    output_size = None
    if output_projection is None:
      dec_cell = tf.contrib.rnn.OutputProjectionWrapper(dec_cell, dec_vocab_size)
      output_size = dec_vocab_size
    dec_init_state = tf.matmul(enc_state, enc2dec_w) + enc2dec_b

    return tf.contrib.legacy_seq2seq.embedding_attention_decoder(
             dec_inputs,
             dec_init_state,
             attention_states,
             dec_cell,
             dec_vocab_size,
             cell_size,
             num_heads = num_heads,
             output_size = output_size,
             output_projection = output_projection,
             feed_previous = feed_previous,
             initial_state_attention = initial_state_attention
           )

if __name__ == '__main__':
  vocab_size = 5000
  batch_size = 64
  cell_size  = 128
  enc_length = 28
  dec_length = 7

  #enc_inputs = [tf.placeholder(tf.int32, [batch_size, cell_size]) for _ in range(enc_length)]
  #dec_inputs = [tf.placeholder(tf.int32, [batch_size, cell_size]) for _ in range(dec_length)]
  enc_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(enc_length)]
  dec_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(dec_length)]

  bidir_attn_seq2seq(
    enc_inputs,
    dec_inputs,
    tf.contrib.rnn.GRUCell,
    cell_size,
    vocab_size,
    vocab_size,
    dtype = tf.float32
  )
