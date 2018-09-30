# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: model.py
@ time: $18-9-29 下午2:43
"""

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from utils.logger import logger


class BiLSTM_CRF(object):
    def __init__(self, FLAG, embeddings, tag2label, vocab, path_dict, config):
        """
        
        :param FLAG: tf.flags params
        :param embedding: word vector
        :param tag2label: BIO tag trans to label dict
        :param vocab: word to id dict 
        :param path_dict: file paths with key
        :param config: setting config
        """
        self.batch_size = FLAG.batch_size
        self.epoch_num = FLAG.epoch
        self.hidden_dim = FLAG.hidden_dim
        self.embeddings = embeddings
        self.CRF = FLAG.CRF
        self.update_embedding = FLAG.update_embedding
        self.dropout_keep_prob = FLAG.dropout
        self.optimizer = FLAG.optimizer
        self.lr = FLAG.lr
        self.clip_grad = FLAG.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = FLAG.shuffle
        self.model_path = path_dict['model_path']
        self.summary_path = path_dict['summary_path']
        self.logger = logger
        self.result_path = path_dict['result_path']
        self.config = config

    def bulid_graph(self):
        """

        :return:
        """
        # set place holder
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        # look up embedding value with id
        with tf.variable_scope('words'):
            _word_embeddings = tf.Variable(self.embeddings, trainable=self.update_embedding,\
                                          dtype=tf.float32, name='_word_embedding')
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids,\
                                                    name='word_embedding')
        # drop out ?
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

        # build bi-lstm layer op
        with tf.variable_scope('bi-lstm'):
            # self.hidden_dim the unit nums in lstm cell
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope('layer_out'):
            """
            Returns an initializer performing "Xavier" initialization for weights.
            This function implements the weight initialization from:
            Xavier Glorot and Yoshua Bengio (2010): Understanding the difficulty of training \
            deep feedforward neural networks. International conference on artificial intelligence \
            and statistics.

            This initializer is designed to keep the scale of the gradients roughly the same \
            in all layers. In uniform distribution this ends up being the range: x = sqrt(6. / (in + out)); \
            [-x, x] and for normal distribution a standard deviation of sqrt(2. / (in + out)) is used.
            """
            W = tf.get_variable(name='W', shape=[2 * self.hidden_dim, self.num_tags],\
                                initializer=tf.contrib.layers.xavier_initializer(),\
                                dtype=tf.float32)

            B = tf.get_variable(name='B', shape=self.num_tags, initializer=tf.zeros_initializer(),\
                                dtype=tf.float32)

            # out_shape[-1]-> step
            out_shape = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            self.logits = tf.reshape((tf.matmul(output, W) + B), [-1, out_shape[1], self.num_tags])

        # build loss op
        if self.CRF:
            # CRF out layer
            with tf.variable_scope('crf_loss'):
                log_likelihood, self.trans = crf_log_likelihood(inputs=self.logits,tag_indices=self.labels,\
                                                                sequence_lengths=self.sequence_lengths)
                self.loss = -tf.reduce_mean(log_likelihood)
        else:
            # with softmax out layer
            with tf.variable_scope('softmax_loss'):

