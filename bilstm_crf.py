# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embedding_size"]
        self.embedding_dim = config["embedding_dim"]
        self.sen_len = config["sen_len"]
        self.tag_size = config["tag_size"]

        self.input_data = tf.placeholder(tf.int32, shape=[None, None], name="input_data")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        with tf.variable_scope("bilstm_crf") as scope:
            self._build_net()

    def _build_net(self):
        word_embeddings = tf.get_variable("word_embeddings", [self.embedding_size, self.embedding_dim])

        input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
        input_embedded = tf.nn.dropout(input_embedded, self.keep_prob)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                         lstm_bw_cell,
                                                                         input_embedded,
                                                                         sequence_length=self.seq_len,
                                                                         dtype=tf.float32,
                                                                         time_major=False)

        bilstm_out = tf.concat([output_fw, output_bw], axis=2)
        bilstm_out = tf.nn.dropout(bilstm_out, self.keep_prob)

        # Fully connected layer.
        W = tf.get_variable(name="W", shape=[2 * self.embedding_dim, self.tag_size],
                            dtype=tf.float32)

        b = tf.get_variable(name="b", shape=[self.tag_size], dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        self.logits = tf.tanh(tf.matmul(tf.reshape(bilstm_out, [-1, 2 * self.embedding_dim]), W) + b)
        self.logits = tf.reshape(self.logits, [-1, 15, self.tag_size])
        self.prediction = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        # sequence_lengths = tf.tile([self.sen_len], [11122])

        # Linear-CRF.
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                   self.labels,
                                                                                   self.seq_len)

        self.loss = tf.reduce_mean(-log_likelihood)

        # Compute the viterbi sequence and score (used for prediction and test time).
        # self.tag, self.viterbi_score = tf.contrib.crf.crf_decode(bilstm_out,
        #                                                          self.transition_params,
        #                                                          sequence_lengths)

        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # Training ops.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
