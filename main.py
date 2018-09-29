# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: main.py
@ time: $18-9-29 上午11:06
"""

import os
import tensorflow as tf
import numpy as np


# 参数设定
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('train_data', 'data', 'path to the train data')
tf.flags.DEFINE_string('test_data', 'data', 'path to the test data')
tf.flags.DEFINE_bool('CRF', True, 'using CRF in top layer, if false using SoftMax')
tf.flags.DEFINE_integer('batch_size', 64, 'size of seg for each batch')
tf.flags.DEFINE_integer('epoch', 40, 'training epoch for each bath')
tf.flags.DEFINE_integer('hidden_dim', 300, 'dims for hidden status')
tf.flags.DEFINE_integer('embedding_dim', 300, 'dims for seg vector')
tf.flags.DEFINE_string('optimizer', 'Adam', 'type of optimizer in traning process, \
                                Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate in training process')
tf.flags.DEFINE_float('clip', 5.0, 'gradient clipping in training process')
tf.flags.DEFINE_float('dropout', 0.5, 'dropout/keep_prob, probability of dropping connection')
tf.flags.DEFINE_bool('update_embedding', True, 'update embedding while training')
tf.flags.DEFINE_string('pretrain_embedding', 'random', 'whether use pretrain embedding file, \
                                                       default generate embedding randomly')
tf.flags.DEFINE_bool('shuffle', True, 'whether shuffle data before epoch')
tf.flags.DEFINE_string('mode', 'train', 'train/test/demo/eval, which mode used for main process')
tf.flags.DEFINE_string('model_identify', '', 'your model identify')

# tf config/ os setting
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# using cpu
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
