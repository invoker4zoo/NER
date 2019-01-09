# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: model.py
@ time: $19-1-9 下午3:38
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

# import rnncell as rnn

class NER_MODEL(object):
    def __init__(self, config, is_train=True):
        """
        :param config: 模型参数字典
        :param is_train:
        """
        self.config = config
        self.is_train = is_train

        self.lr = config["lr"]
        self.clip = config["clip"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        # 0/1/2/3 seg id
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        # Returns an initializer performing "Xavier" initialization for embedding layer weights.
        self.initializer = initializers.xavier_initializer()
        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        # dropout prob = 1 - self.dropout
        self.keep_dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
        # origin
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        # format length batch: [char_length1,char_length2,char_length3,..]
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        # self.length = self.batch_size * self.num_steps
        # self.length = tf.cast(self.length, tf.int32)