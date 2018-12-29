# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: util.py
@ time: $18-12-29 上午10:50
"""
import sys
from logger import logger
import re


######################
##  process funtion
def load_sentence_file(path, zeros, lower):
    """
    载入句子文件
    :param path: path of file
    :param zeros: whether trans num to zero
    :param lower: whether lower char
    line format: 测 O/
    line.split()
    :return:
    """
    try:
        sentences = list()
        sentence = list()
        with open(path, 'rb') as f:
            lines = f.readlines()
        for index, line in enumerate(lines):
            line = trans_num_to_zero(line.rstrip()) if zeros else line.rstrip()
            if line:
                if line[0] == ' ':
                    line = '&' + line[1:]
                    sentence.append(line.split())
                else:
                    sentence.append(line.split())
            else:
                if len(sentence)>0:
                    sentences.append(sentence)
                    sentence = []
        return sentences
    except Exception, e:
        logger.error('load sentence file failed for %s' % str(e))
        return []


######################
##  tool funtion
def trans_num_to_zero(str):
    """
    将字符串中的数字全部转为0
    :param str:
    :return:
    """
    return re.sub('\d', '0', str)