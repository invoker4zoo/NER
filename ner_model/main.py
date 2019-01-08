# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: main.py
@ time: $18-12-24 下午5:28
"""

import tensorflow as tf
import os
from tool.util import *
from tool.logger import logger
import pickle


flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     False,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt_biLSTM",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "data/_maps.pkl",     "file for chars index mapping")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "data/vec.txt",  "pre_trained embedding file")
flags.DEFINE_string("train_file",   "data/example.train",  "train data file")
flags.DEFINE_string("dev_file",     "data/example.dev",    "dev data file")
flags.DEFINE_string("test_file",    "data/example.test",   "test data file")


flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
FLAGS = tf.app.flags.FLAGS

# # build file path
# flags.DEFINE_string("emb_file",     os.path.join("data", FLAGS.emb_file_name),  "Path for pre_trained embedding")
# flags.DEFINE_string("train_file",   os.path.join("data", FLAGS.train_file_name),  "Path for train data")
# flags.DEFINE_string("dev_file",     os.path.join("data", FLAGS.dev_file_name),    "Path for dev data")
# flags.DEFINE_string("test_file",    os.path.join("data", FLAGS.test_file_name),   "Path for test data")

def build_config(char_to_id, tag_to_id):
    """
    建立graph中需求的config字典
    :param char_to_id: 字符转向id的字典
    :param tag_to_id: 标签转向id的字典
    config key
    model_type: 模型使用方法 idcnn/bilstm
    num_chars: 字符字典的长度
    char_dim: embedding 层转换后的字符向量维度
    num_tags: 标签字典的长度
    seg_dim: embedding 层转换后的分词情况向量维度
    :return:
    """
    config = dict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def string_pre_process_for_train():
    """
    将文本转换为模型输入的前处理流程

    :return:
    """
    try:
        train_sentences = load_sentence_file(FLAGS.train_file, FLAGS.zeros)
        dev_sentences = load_sentence_file(FLAGS.dev_file, FLAGS.zeros)
        test_sentences = load_sentence_file(FLAGS.test_file, FLAGS.zeros)
        # change tag schema in sentence
        trans_tag_schema(train_sentences, FLAGS.tag_schema)
        trans_tag_schema(test_sentences, FLAGS.tag_schema)
        # loading/writing mapping file
        if not os.path.isfile(FLAGS.map_file):
            logger.info('mapping file does not exist, create mapping file')
            if FLAGS.pre_emb:
                pass
            else:
                char_count_dic, id_to_char, char_to_id = char_mapping(train_sentences, FLAGS.lower)
            tag_count_dic, id_to_tag, tag_to_id = tag_mapping(train_sentences)
            with open(FLAGS.map_file, 'wb') as f:
                # notice pickle file format with py2 and py3
                pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
        else:
            logger.info('loading mapping file')
            with open(FLAGS.map_file, 'rb') as f:
                char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

        # prepare model data


    except Exception, e:
        logger.error('pre-process for train string failed for %s' % str(e))

