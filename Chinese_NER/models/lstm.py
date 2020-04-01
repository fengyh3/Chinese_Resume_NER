import tensorflow as tf
from .util import get_next_batch, get_batches
import numpy as np
import sys
sys.path.append('../')
from evaluate import Metrics

class BiLSTM(object):
    def __init__(self, vocab_size, tag_size, batch_size = 64, lr = 0.001, iteration = 20, hidden_size = 128, embedding_size = 128):
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.batch_size = batch_size
        self.lr = lr
        self.iteration = iteration
        self.hidden_size = hidden_size
        #self.seq_len = 100
        self.embedding_size = embedding_size
        self.word_embedding = tf.Variable(initial_value = tf.random_normal(shape=[vocab_size, embedding_size]), trainable = True)

    def add_placeholder(self):
        self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'input_x')
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'input_y')
        self.seq_lengths = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_lengths')
        self.dropout = tf.placeholder(dtype = tf.float32, shape = [], name = 'dropout')


    def operation(self):
        with tf.name_scope('embedding'):
            chars_vector = tf.nn.embedding_lookup(self.word_embedding, ids = self.input_x, name = 'chars_vector')
            chars_vector = tf.nn.dropout(chars_vector, self.dropout)
        with tf.name_scope('Bi-LSTM'):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name = 'fw_lstm')
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name = 'bw_lstm')
            output, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, inputs = chars_vector, sequence_length = self.seq_lengths, dtype = tf.float32)
            fw_output = output[0]
            bw_output = output[1]
            concat = tf.concat([fw_output, bw_output], -1, name = 'Bi-LSTM-concat')
            concat = tf.nn.dropout(concat, self.dropout)
            s = tf.shape(concat)
            concat = tf.reshape(concat, shape = [-1, 2 * self.hidden_size])

        with tf.name_scope('projection'):
            W = tf.get_variable('W', dtype = tf.float32, shape = [2 * self.hidden_size, self.tag_size])
            b = tf.get_variable('b', dtype = tf.float32, shape = [self.tag_size])
            pred = tf.nn.dropout(tf.matmul(concat, W) + b, self.dropout)
            self.logit = tf.reshape(pred, shape = [-1, s[1], self.tag_size])

    def loss_op(self):
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logit, labels = self.input_y)
            mask = tf.sequence_mask(self.seq_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

    def optimize(self):
        with tf.name_scope('optimize'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def pred_batch(self):
        self.pred = tf.cast(tf.argmax(self.logit, axis = -1), dtype = tf.int32)


    def train(self, train_x, train_y, dev_x, dev_y, word2id, tag2id, dropout):
        self.add_placeholder()
        self.operation()
        self.loss_op()
        self.pred_batch()
        self.optimize()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        x, y, seq = get_batches(train_x, train_y, word2id, tag2id, self.batch_size)
        for i in range(self.iteration):
            for j in range(len(x)):
                _, loss, pred_labels = self.sess.run([self.optimizer, self.loss, self.pred], feed_dict = {self.input_x:x[j], self.input_y:y[j], self.seq_lengths:seq[j], self.dropout:dropout})
            #self.dev_test(dev_x, dev_y, word2id, tag2id)


    def dev_test(self, dev_x, dev_y, word2id, tag2id):
        batches_x, batches_y, batches_seq_len = get_batches(dev_x, dev_y, word2id, tag2id, self.batch_size)
        pred_lists = []
        labels = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i in range(len(batches_x)):
            pred_labels, loss = self.sess.run([self.pred, self.loss], feed_dict = {self.input_x:batches_x[i], self.input_y:batches_y[i], self.seq_lengths:batches_seq_len[i], self.dropout:1.0})
            for j in range(len(pred_labels)):
                for k in range(batches_seq_len[i][j]):
                    pred_lists.append(id2tag[pred_labels[j][k]])
                    labels.append(id2tag[batches_y[i][j][k]])
        metrics = Metrics(labels, pred_lists)
        metrics.report_scores()

    def close_sess(self):
        self.sess.close()