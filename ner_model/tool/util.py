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
# use jieba as seg cut model first
import jieba
import math
import random
import os
import json


######################
##  process funtion
def load_sentence_file(path, zeros, lower=False):
    """
    载入句子文件
    :param path: path of file
    :param zeros: bool whether trans num to zero
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
            line = line.lower() if lower else line
            if line:
                if line[0] == ' ':
                    line = '$' + line[1:]
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


def trans_tag_schema(sentences, tag_schema):
    """
    将标准模式转为统一的模式
    :param sentences: 载入的句子列表 [[[char, tag],...],]
    :param tag_schema: bio/bioes
    :return:
    """
    try:
        for index, sentence in enumerate(sentences):
            tags = [item[-1] for item in sentence]
            if not check_tag_schema(tags):
                error_str = ''.join([word[0] for word, _ in sentence])
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %d:\n%s' % (index, error_str))
            if tag_schema == 'iob':
                for word, new_tag in zip(sentence, tags):
                    word[-1] = new_tag
            elif tag_schema == 'iobes':
                new_tags = iob_to_iobes(tags)
                for word, new_tag in zip(sentence, new_tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')
    except Exception, e:
        logger.error('trans tag schema failed for %s' % str(e))


def char_mapping(sentences, lower):
    """
    获取字符字典
    :param sentences:[[[char1, tag1], [char2, tag2], ...], []]
    :param lower:
    :return: count dic, char_to_id, id_to_char
    """
    try:
        # char list
        chars = [[char[0].lower() if lower else char[0] for char in sentence] for sentence in sentences]
        count_dic = create_dict(chars)
        count_dic["<PAD>"] = 10000001
        count_dic["<UNK>"] = 10000000
        id_to_char, char_to_id = create_mapping(count_dic)
        return count_dic, id_to_char, char_to_id
    except Exception, e:
        logger.error('char mapping trans failed for %s' %str(e))
        return {}, {}, {}


def tag_mapping(sentences):
    """
    获取标签的字典映射
    :param sentences:
    :return:
    """
    try:
        tags = [[items[-1] for items in sentence] for sentence in sentences]
        count_dic = create_dict(tags)
        id_to_tag, tag_to_id = create_mapping(count_dic)
        return count_dic, id_to_tag, tag_to_id
    except Exception, e:
        logger.error('tag mapping trans failed for %s' %str(e))
        return {}, {}, {}


def prepare_model_data(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    建立模型需求输入数据结构
    :param sentences:
    :param char_to_id:
    :param tag_to_id:
    :return: data  --- [[char_list, char_id_list, seg_id_list, tags_id_list],[]]
             seg_id_list example: [X/XX/XXX/XXXX] -> [0 /1 3 /1 2 3 /1 2 2 3]
    """
    try:
        def lower(char):
            return char.lower() if lower else char
        no_train_tag_index = tag_to_id['O']
        model_data = list()
        for sentence in sentences:
            char_list = [item[0] for item in sentence]
            char_id_list = [char_to_id[lower(char) if lower(char) else "<UNK>"] for char in char_list]
            sentence_string = "".join(char_list)
            seg_id_list = get_seg_feature(sentence_string)
            tags = [item[-1] for item in sentence]
            if train:
                tag_id_list = [tag_to_id[tag] for tag in tags]
            else:
                tag_id_list = [no_train_tag_index] * len(tags)
            # append setence data
            model_data.append([char_list, char_id_list, seg_id_list, tag_id_list])
        return model_data
    except Exception, e:
        logger.error('prepare model data failed for %s' % str(e))
        return None

class BatchManager(object):
    """
    将处理后的data转化为batch iter的复用类
    """
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

######################
##  tool funtion
def trans_num_to_zero(str):
    """
    将字符串中的数字全部转为0
    :param str:
    :return:
    """
    return re.sub('\d', '0', str)

def check_tag_schema(tags):
    """
    检测标签体系是否符合BIO体系
    :param tags:
    :return:
    """
    for index, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif index == 0 or tags[index - 1] == 'O':  # conversion IOB1 to IOB2
            tags[index] = 'B' + tag[1:]
        elif tags[index - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[index] = 'B' + tag[1:]
    return True


def iob_to_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def create_dict(chars):
    """
    将字符串列表转为计数字典
    :param chars:[[char1, char2, ...], [char3, char4, ...]]
    :return dic
    """
    dic = dict()
    for char_list in chars:
        for char in char_list:
            if char in dic.keys():
                dic[char] += 1
            else:
                dic[char] = 1
    return dic


def create_mapping(count_dic):
    """
    将计数字典转为多个id-char映射字典
    :param count_dic: 字符串计数字典
    :return: id_to_char, char_to_id
    """
    sorted_items = sorted(count_dic.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {index: item for index, item in enumerate(sorted_items)}
    item_to_id = {item: index for index, item in id_to_item.items()}
    return id_to_item, item_to_id


def get_seg_feature(str):
    """
    对字符串分词后转换为分词id
    :param str:
    :return:
    """
    seg_id_list = list()
    for seg in jieba.cut(str):
        if len(seg) == 1:
            seg_id_list.append(0)
        else:
            seg_id = [2] * len(seg)
            seg_id[0] = 1
            seg_id[-1] = 3
            seg_id_list.extend(seg_id)
    return seg_id_list


def make_path(FLAGS):
    """
    建立file_path
    :param FLAGS: tensorflow FLAGS
    :return:
    """
    if not os.path.isdir(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    if not os.path.isdir(FLAGS.ckpt_path):
        os.makedirs(FLAGS.ckpt_path)
    # represent with logger
    # if not os.path.isdir("log"):
    #     os.makedirs("log")


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, 'rb') as f:
        return json.loads(f.read())


