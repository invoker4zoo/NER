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

import math
import time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from utils.logger import logger
from utils.data_script import generate_batch, pad_sequences


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
        # initial params
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
        self.model_path = path_dict.get('model_path')
        self.summary_path = path_dict.get('summary_path')
        self.logger = logger
        self.result_path = path_dict.get('result_path')
        self.config = config
        # build graph
        self.build_graph()

    def build_graph(self):
        """
        创建计算图
        :return:
        """
        # set place holder
        # word_ids, labels bi-lstm input,补齐长度后序列
        # sequence_lengths 原始句子长度 bi-lstm params sequence_length
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
                log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,tag_indices=self.labels,\
                                                                sequence_lengths=self.sequence_lengths)
                self.loss = -tf.reduce_mean(log_likelihood)
        else:
            # with softmax out layer
            with tf.variable_scope('softmax_loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)
                self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
                self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
        tf.summary.scalar("loss", self.loss)

        # build train graph
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g[0], -self.clip_grad, self.clip_grad), g[1]] for g in grads_and_vars]
            # global_step variable will increase 1 by every calculation
            self.train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

        # init variable
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """
        过程存储
        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def build_feed_dict(self, sentence_list, labels_list=None, lr=None, drop_out=None):
        """
        创建训练中的feed_dict
        fill place_holder
        :param sentence_list: [[id,id...],]
        :param labels_list:
        :param lr: learning rate
        :param drop_out:
        :return:feed_dict
        """
        feed_dict = dict()
        _sentence_list, sentence_len_list = pad_sequences(sentence_list, pad_mark=0)
        feed_dict[self.word_ids] = _sentence_list
        feed_dict[self.sequence_lengths] = sentence_len_list
        if labels_list:
            _labels_list, _ = pad_sequences(labels_list, pad_mark=0)
            feed_dict[self.labels] = _labels_list
        if lr:
            feed_dict[self.lr_pl] = lr
        if drop_out:
            feed_dict[self.dropout_pl] = drop_out
        return feed_dict

    def train(self, train_data):
        """
        训练主函数
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        print '*' * 30
        logger.info('begin training....')
        with tf.Session(config=self.config) as sess:
            # 初始化参数，添加summary
            sess.run(self.init_op)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                logger.info('begin No.%d epoch process' % epoch)
                # num_batches = len(train_data) // self.batch_size + 1
                num_batches = math.ceil(len(train_data) / float(self.batch_size))
                batches = generate_batch(train_data, self.batch_size, self.vocab,\
                                         self.tag2label, shuffle=self.shuffle)
                start_time = time.time()
                batch_begin_time = time.time()
                for step, (sentence_list, labels_list) in enumerate(batches):
                    feed_dict = self.build_feed_dict(sentence_list, labels_list, self.lr, self.dropout_keep_prob)
                    _, train_loss, train_summary, _global_step = sess.run([self.train_op, self.loss, self.merged, self.global_step],\
                              feed_dict=feed_dict)
                    batch_using_time = time.time() - batch_begin_time
                    batch_begin_time = time.time()
                    logger.info('No.%d batch / %d batches, batch training time %.2f s' % (step + 1, num_batches, batch_using_time))
                    global_step = epoch * num_batches + step + 1
                    if global_step % 300 == 0 or step + 1 == num_batches:
                        using_time = time.time() - start_time
                        start_time = time.time()
                        logger.info('epoch {}, step {}, loss {:.4}, global step {}, training time {}s'.format(epoch+1,\
                                                                        step+1, train_loss, global_step, using_time))

                    self.file_writer.add_summary(train_summary, global_step)

                    if step + 1 == num_batches:
                        saver.save(sess, self.model_path, global_step=global_step)

    def _prediction_one_batch(self, sess, sentence_list):
        """
        在计算图中跑一个batch,得到标签列表
        :param sess:
        :param sentence_list: [[],[]]
        :return: label_list [[],[]]
        """
        feed_dict = self.build_feed_dict(sentence_list, drop_out=1.0)
        seq_len_list = [len(sentence) for sentence in sentence_list]
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list

    def prediction(self, sess, demo_data):
        """
        预测demo_data的标签
        :param sess: tf.sess
        :param demo_data:[([seg,seg],[label, label]),...] /[(seg_list, label_list)]
        :return:tag_list 标签列表
        """
        # feed_dict = self.build_feed_dict(demo_data, drop_out=1.0)
        label_list = list()
        for sentence_list, labels_list in generate_batch(demo_data, self.batch_size, self.vocab,\
                                               self.tag2label, shuffle=self.shuffle):
            _labels = self._prediction_one_batch(sess, sentence_list)
            labels_list.append(_labels)
        label2tag = dict()
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag_list = [label2tag[label] for label in label_list[0]]
        return tag_list

    def evaluate(self, sess, eval_data):
        """
        
        :param sess:
        :param eval_data:
        :return:
        """
