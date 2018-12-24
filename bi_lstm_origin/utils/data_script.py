# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: data_script.py
@ time: $18-10-8 上午10:00
"""

import sys
import os
import random

## tags dict, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def sentence2id(sentence, word2id):
    """
    文字转编号
    :param sentence: ''
    :param word2id: vocab
    :return:
    """
    sentence_id = []
    for word in sentence:
        # if word.isdigit():
        #     word = '<NUM>'
        try:
            int(word)
            word = '<NUM>'
        except:
            pass
        if ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def generate_batch(data, batch_size, vocab, tag2label, shuffle=False):
    """
    由训练数据生成batch
    :param data: data type [(sentence, labels), (sentence, labels)...]
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    sentence_list, labels_list = list(), list()
    for _sentence, _labels in data:
        _sentence = sentence2id(_sentence, vocab)
        _labels = [tag2label[label] for label in _labels]
        if len(sentence_list) == batch_size:
            yield sentence_list, labels_list
            sentence_list, labels_list = list(), list()
        else:
            sentence_list.append(_sentence)
            labels_list.append(_labels)
    if len(sentence_list):
        yield sentence_list, labels_list

        
def pad_sequences(seq_list, pad_mark=0):
    """
    用于补齐序列长度，使序列列表中的每一个序列长度都与最长序列相同
    :param seq:
    :param pad_mark:
    :return: seq_list 补齐后的序列列表
             seq_len_list 原始序列列表中每一个序列的长度列表
    """
    max_len = max(map(lambda x : len(x), seq_list))
    seq_list, seq_len_list = [], []
    for seq in seq_list:
        seq = list(seq)
        _seq = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(_seq)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list