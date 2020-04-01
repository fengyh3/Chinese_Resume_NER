from .lstm import BiLSTM
import tensorflow as tf
from tensorflow.contrib import crf
from .util import get_batches
import sys
sys.path.append('../')
from evaluate import Metrics


class BiLSTM_CRF(object):
    def __init__(self, vocab_size, tag_size, batch_size = 64, lr = 0.001, iteration = 30, hidden_size = 128, embedding_size = 128):
        tf.reset_default_graph()
        self.bilstm = BiLSTM(vocab_size, tag_size, batch_size, lr, iteration, hidden_size, embedding_size)


    def CRF_layer(self):
        self.logit = self.bilstm.logit
        with tf.name_scope('crf'):
            log_likelihood_, self.transition = crf.crf_log_likelihood(self.logit, self.bilstm.input_y, self.bilstm.seq_lengths)
            self.cost = -tf.reduce_mean(log_likelihood_)

    def optimize(self):
        with tf.name_scope('crf_optimize'):
            self.optimizer = tf.train.AdamOptimizer(self.bilstm.lr).minimize(self.cost)


    def train(self, train_x, train_y, dev_x, dev_y, word2id, tag2id, dropout):
        self.bilstm.add_placeholder()
        self.bilstm.operation()
        self.CRF_layer()
        self.optimize()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        x, y, seqs = get_batches(train_x, train_y, word2id, tag2id, self.bilstm.batch_size)
        for i in range(self.bilstm.iteration):
            for j in range(len(x)):
                _, loss = self.sess.run([self.optimizer, self.cost], feed_dict = {self.bilstm.input_x:x[j], self.bilstm.input_y:y[j], self.bilstm.seq_lengths:seqs[j], self.bilstm.dropout:dropout})
            #self.dev_test(dev_x, dev_y, word2id, tag2id)
                

    def pred_labels(self, x, y, seqs):
        scores, transition_matrix = self.sess.run([self.logit, self.transition], feed_dict = {self.bilstm.input_x:x, self.bilstm.input_y:y, self.bilstm.seq_lengths:seqs, self.bilstm.dropout:1.0})
        labels  = []
        for i in range(scores.shape[0]):
            label, _ = crf.viterbi_decode(scores[i], transition_params = transition_matrix)

            labels.append(label)
        return labels

    def dev_test(self, dev_x, dev_y, word2id, tag2id):
        batches_x, batches_y, batches_seq_len = get_batches(dev_x, dev_y, word2id, tag2id, self.bilstm.batch_size)
        pred_lists = []
        labels = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())

        for i in range(len(batches_x)):
            pred_labels = self.pred_labels(batches_x[i], batches_y[i], batches_seq_len[i])
            for j in range(len(pred_labels)):
                for k in range(batches_seq_len[i][j]):
                    pred_lists.append(id2tag[pred_labels[j][k]])
                    labels.append(id2tag[batches_y[i][j][k]])
        metrics = Metrics(labels, pred_lists)
        metrics.report_scores()

    def close_sess(self):
        self.sess.close()