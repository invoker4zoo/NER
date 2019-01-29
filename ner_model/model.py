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
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tool.util import iob_to_iobes, result_to_json

# import rnncell as rnn

# class NER_MODEL(object):
#     def __init__(self, config, is_train=True):
#         """
#         :param config: 模型参数字典
#         :param is_train:
#         """
#         self.config = config
#         self.is_train = is_train
#         # initial graph
#
#         self.lr = self.config["lr"]
#         self.clip = self.config["clip"]
#         self.char_dim = self.config["char_dim"]
#         self.lstm_dim = self.config["lstm_dim"]
#         self.seg_dim = self.config["seg_dim"]
#
#         self.num_tags = self.config["num_tags"]
#         self.num_chars = self.config["num_chars"]
#         # 0/1/2/3 seg id
#         self.num_segs = 4
#         self.global_step = tf.Variable(0, trainable=False)
#
#         # Returns an initializer performing "Xavier" initialization for embedding layer weights.
#         self.initializer = initializers.xavier_initializer()
#
#         # add placeholders for the model
#         self.char_inputs = tf.placeholder(dtype=tf.int32,
#                                           shape=[None, None],
#                                           name="ChatInputs")
#         self.seg_inputs = tf.placeholder(dtype=tf.int32,
#                                          shape=[None, None],
#                                          name="SegInputs")
#         self.targets = tf.placeholder(dtype=tf.int32,
#                                       shape=[None, None],
#                                       name="Targets")
#         # dropout keep prob
#         # dropout prob = 1 - self.dropout
#         self.keep_dropout = tf.placeholder(dtype=tf.float32,
#                                       name="Dropout")
#         # origin
#         used = tf.sign(tf.abs(self.char_inputs))
#         length = tf.reduce_sum(used, reduction_indices=1)
#         # format length batch: [char_length1,char_length2,char_length3,..]
#         # real length of each sequence / without padding char tag
#         self.lengths = tf.cast(length, tf.int32)
#         self.batch_size = tf.shape(self.char_inputs)[0]
#         self.num_steps = tf.shape(self.char_inputs)[-1]
#         # self.length = self.batch_size * self.num_steps
#         # self.length = tf.cast(self.length, tf.int32)
#
#         # model type bilstm/idcnn
#         self.model_type = self.config['model_type']
#         # initial parameters for idcnn model
#         self.layers = [
#             {
#                 'dilation': 1
#             },
#             {
#                 'dilation': 1
#             },
#             {
#                 'dilation': 2
#             },
#         ]
#         self.filter_width = 3
#         self.num_filter = self.lstm_dim
#         self.embedding_dim = self.char_dim + self.seg_dim
#         self.repeat_times = 4
#         self.cnn_output_width = 0
#
#         # saver of the model
#         self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
#
#
#         # build self.embedding
#         self.embedding_layer_op()
#         if self.model_type == 'idcnn':
#             # build self.model_output and self.logits with idcnn model
#             self.idcnn_layer_op()
#             self.model_layer_idcnn_op()
#         elif self.model_type == 'bilstm':
#             # build self.model_output and self.logits with bilstm model
#             self.bilstm_layer_op()
#             self.model_layer_bilstm_op()
#         else:
#             raise KeyError
#         # build self.loss calculate model loss
#         self.loss_op()
#         # build self.train_op
#         self.train_process_op()
#
#
#     def embedding_layer_op(self):
#         """
#         self.embedding:[[char_embedding, seg_embedding]]
#         :return:
#         """
#         _embedding = list()
#         with tf.variable_scope('char_embedding'), tf.device('/cpu:0'):
#             self.char_lookup = tf.get_variable(
#                 name='char_embedding',
#                 shape=[self.num_chars, self.char_dim],
#                 initializer=self.initializer
#             )
#             _embedding.append(tf.nn.embedding_lookup(self.char_lookup, self.char_inputs))
#             # check seg_dim exist
#             if self.config.get('seg_dim'):
#                 with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
#                     self.seg_lookup = tf.get_variable(
#                         name='seg_embedding',
#                         shape=[self.num_segs, self.seg_dim],
#                         initializer=self.initializer
#                     )
#                     _embedding.append(tf.nn.embedding_lookup(self.seg_lookup, self.seg_inputs))
#             # concat char and seg embedding
#             self.embedding = tf.concat(_embedding, axis=-1)
#
#     def bilstm_layer_op(self):
#         """
#         build bi-lstm layer op
#         :return:
#         """
#         model_input = tf.nn.dropout(self.embedding, self.keep_dropout)
#         with tf.variable_scope('char_BiLSTM'):
#             cell_fw = LSTMCell(self.lstm_dim)
#             cell_bw = LSTMCell(self.lstm_dim)
#             (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
#                 cell_fw=cell_fw,
#                 cell_bw=cell_bw,
#                 inputs=model_input,
#                 sequence_length=self.lengths,
#                 dtype=tf.float32)
#             output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
#             self.model_output = output
#
#     def idcnn_layer_op(self):
#         """
#         build idcnn layer op
#         :param model_inputs: [batch_size, num_steps, emb_size]
#         :return: [batch_size, num_steps, cnn_output_width]
#         """
#         model_input = tf.nn.dropout(self.embedding, self.keep_dropout)
#         model_input = tf.expand_dims(model_input, 1)
#         reuse = False
#         if not self.is_train:
#             reuse = True
#         # shape = [1, self.filter_width, self.embedding_dim,
#         #          self.num_filter]
#         #
#         # # print(shape)
#         with tf.variable_scope("idcnn"):
#             filter_weights = tf.get_variable(
#                 "idcnn_filter",
#                 shape=[1, self.filter_width, self.embedding_dim,
#                        self.num_filter],
#                 initializer=self.initializer)
#
#             """
#             shape of input = [batch, in_height, in_width, in_channels]
#             shape of filter = [filter_height, filter_width, in_channels, out_channels]
#             """
#             layerInput = tf.nn.conv2d(model_input,
#                                       filter_weights,
#                                       strides=[1, 1, 1, 1],
#                                       padding="SAME",
#                                       name="init_layer")
#             finalOutFromLayers = []
#             totalWidthForLastDim = 0
#             for j in range(self.repeat_times):
#                 for i in range(len(self.layers)):
#                     dilation = self.layers[i]['dilation']
#                     isLast = True if i == (len(self.layers) - 1) else False
#                     with tf.variable_scope("atrous-conv-layer-%d" % i,
#                                            reuse=tf.AUTO_REUSE):
#                         w = tf.get_variable(
#                             "filterW",
#                             shape=[1, self.filter_width, self.num_filter,
#                                    self.num_filter],
#                             initializer=tf.contrib.layers.xavier_initializer())
#                         b = tf.get_variable("filterB", shape=[self.num_filter])
#                         # 膨胀卷积atrous_conv2d
#                         conv = tf.nn.atrous_conv2d(layerInput,
#                                                    w,
#                                                    rate=dilation,
#                                                    padding="SAME")
#                         conv = tf.nn.bias_add(conv, b)
#                         conv = tf.nn.relu(conv)
#                         if isLast:
#                             finalOutFromLayers.append(conv)
#                             totalWidthForLastDim += self.num_filter
#                         layerInput = conv
#             finalOut = tf.concat(axis=3, values=finalOutFromLayers)
#             keepProb = 1.0 if reuse else 0.5
#             finalOut = tf.nn.dropout(finalOut, keepProb)
#
#             finalOut = tf.squeeze(finalOut, [1])
#             finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
#             self.cnn_output_width = totalWidthForLastDim
#             self.model_output = finalOut
#
#     def model_layer_bilstm_op(self):
#         """
#         bilstm model add hidden layer before logits layer
#         :return: [batch_size, num_steps, num_tags]
#         """
#         with tf.variable_scope("project"):
#             # hidden layer
#             with tf.variable_scope("hidden"):
#                 W = tf.get_variable("W", shape=[self.lstm_dim * 2, self.lstm_dim],
#                                     dtype=tf.float32, initializer=self.initializer)
#
#                 b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
#                                     initializer=tf.zeros_initializer())
#                 output = tf.reshape(self.model_output, shape=[-1, self.lstm_dim * 2])
#                 hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
#
#             # project to score of tags
#             with tf.variable_scope("logits"):
#                 W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
#                                     dtype=tf.float32, initializer=self.initializer)
#
#                 b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
#                                     initializer=tf.zeros_initializer())
#                 pred = tf.nn.xw_plus_b(hidden, W, b)
#             self.logits = tf.reshape(pred, [-1, self.num_steps, self.num_tags])
#
#     def model_layer_idcnn_op(self):
#         """
#         :param
#         :return: [batch_size, num_steps, num_tags]
#         """
#         with tf.variable_scope("project"):
#             # project to score of tags
#             with tf.variable_scope("logits"):
#                 W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
#                                     dtype=tf.float32, initializer=self.initializer)
#                 # different initialzer
#                 b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_tags]))
#
#                 pred = tf.nn.xw_plus_b(self.model_output, W, b)
#
#             self.logits = tf.reshape(pred, [-1, self.num_steps, self.num_tags])
#
#     def loss_op(self):
#         """
#         calculate model loss with crf
#         :return:
#         """
#         with tf.variable_scope('crf_loss'):
#             # increase num_step and num_tags shape
#             # small = -1000.0
#             # # pad logits for crf loss
#             # start_logits = tf.concat(
#             #     [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
#             #     axis=-1)
#             # pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
#             # logits = tf.concat([self.logits, pad_logits], axis=-1)
#             # logits = tf.concat([start_logits, logits], axis=1)
#             # targets = tf.concat(
#             #     [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
#             # build trans matrix
#             # self.trans = tf.get_variable(
#             #     "transitions",
#             #     shape=[self.num_tags + 1, self.num_tags + 1],
#             #     initializer=self.initializer)
#
#             # original crf model
#             # self.trans = tf.get_variable('trans_matrix', shape=[self.num_tags, self.num_tags], \
#             #                              initializer=self.initializer)
#             # log_likelihood, self.trans = crf_log_likelihood(
#             #     inputs=self.logits,
#             #     tag_indices=self.targets,
#             #     transition_params=self.trans,
#             #     sequence_lengths=self.lengths)
#             # self.loss = tf.reduce_mean(-log_likelihood)
#             # increase shape model
#
#             small = -1000.0
#             # pad logits for crf loss
#             start_logits = tf.concat(
#                 [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
#                 axis=-1)
#             pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
#             logits = tf.concat([self.logits, pad_logits], axis=-1)
#             logits = tf.concat([start_logits, logits], axis=1)
#             targets = tf.concat(
#                 [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
#
#             self.trans = tf.get_variable(
#                 "transitions",
#                 shape=[self.num_tags + 1, self.num_tags + 1],
#                 initializer=self.initializer)
#             log_likelihood, self.trans = crf_log_likelihood(
#                 inputs=logits,
#                 tag_indices=targets,
#                 transition_params=self.trans,
#                 sequence_lengths=self.lengths + 1)
#             self.loss = tf.reduce_mean(-log_likelihood)
#
#     def train_process_op(self):
#         """
#         build train op with loss
#         :return:
#         """
#         # build optimizer
#         with tf.variable_scope("optimizer"):
#             optimizer = self.config["optimizer"]
#             if optimizer == "sgd":
#                 self.opt = tf.train.GradientDescentOptimizer(self.lr)
#             elif optimizer == "adam":
#                 self.opt = tf.train.AdamOptimizer(self.lr)
#             elif optimizer == "adgrad":
#                 self.opt = tf.train.AdagradOptimizer(self.lr)
#             else:
#                 raise KeyError
#         # self.opt.minimize()
#         # apply grad clip to avoid gradient explosion
#         grads_vars = self.opt.compute_gradients(self.loss)
#         capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
#                              for g, v in grads_vars]
#         self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
#
#     def create_feed_dict(self, is_train, batch):
#         """
#         :param is_train: bool, True for train batch
#         :param batch: list train/evaluate data
#         :return: structured data to feed
#         """
#         _, chars, segs, tags = batch
#         feed_dict = {
#             self.char_inputs: np.asarray(chars),
#             self.seg_inputs: np.asarray(segs),
#             self.keep_dropout: 1.0,
#         }
#         if is_train:
#             feed_dict[self.targets] = np.asarray(tags)
#             feed_dict[self.keep_dropout] = self.config["dropout_keep"]
#         return feed_dict
#
#     def run_step(self, sess, is_train, batch):
#         """
#
#         :param sess: tf session
#         :param is_train: bool
#         :param batch: batch data [[char_list, char_id_list, seg_id_list, tags_id_list],[]]
#         :return: batch result, loss of the batch or logits
#         """
#         feed_dict = self.create_feed_dict(is_train, batch)
#         # train part need loss / evaluate need logits to calculate tags list with viterbi-decode method
#         if is_train:
#             global_step, loss, _ = sess.run(
#                 [self.global_step, self.loss, self.train_op],
#                 feed_dict)
#             return global_step, loss
#         else:
#             lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
#             return lengths, logits
#
#     def decode(self, logits, lengths, matrix):
#         """
#         decode tag list with logtis using viterbi-decode method
#         :param logits: [batch_size, num_steps, num_tags]
#         :param lengths: [batch_size]real length of each sequence
#         :param matrix: transaction matrix in crf
#         :return:
#         """
#         # inference final labels usa viterbi Algorithm
#         # tags = []
#         # small = -1000.0
#         # for score, length in zip(logits, lengths):
#         #     score = score[:length]
#         #     pad = small * np.ones([length, 1])
#         #     logits = np.concatenate([score, pad], axis=1)
#         #     viterbi, viterbi_score = viterbi_decode(logits, matrix)
#         #
#         #     tags.append(viterbi)
#         # return tags
#         # increase shape for saved model
#         tags = []
#         small = -1000.0
#         start = np.asarray([[small] * self.num_tags + [0]])
#         for score, length in zip(logits, lengths):
#             score = score[:length]
#             pad = small * np.ones([length, 1])
#             logits = np.concatenate([score, pad], axis=1)
#             logits = np.concatenate([start, logits], axis=0)
#             path, _ = viterbi_decode(logits, matrix)
#
#             tags.append(path[1:])
#         return tags
#
#     def evaluate(self, sess, data_manager, id_to_tag):
#         """
#         :param sess: session  to run the model
#         :param data: list of data
#         :param id_to_tag: index to tag name
#         :return: evaluate result
#         """
#         results = []
#         trans = self.trans.eval()
#         for batch in data_manager.iter_batch():
#             strings = batch[0]
#             tags = batch[-1]
#             lengths, scores = self.run_step(sess, False, batch)
#             batch_paths = self.decode(scores, lengths, trans)
#             for i in range(len(strings)):
#                 result = []
#                 string = strings[i][:lengths[i]]
#                 gold = iob_to_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
#                 pred = iob_to_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
#                 for char, gold, pred in zip(string, gold, pred):
#                     result.append(" ".join([char, gold, pred]))
#                 results.append(result)
#         return results
#
#     def evaluate_line(self, sess, inputs, id_to_tag):
#         trans = self.trans.eval()
#         # trans = self.trans
#         lengths, scores = self.run_step(sess, False, inputs)
#         batch_paths = self.decode(scores, lengths, trans)
#         tags = [id_to_tag[idx] for idx in batch_paths[0]]
#         return result_to_json(inputs[0][0], tags)

class NER_MODEL(object):
    def __init__(self, config, is_train=True):

        self.config = config
        self.is_train = is_train

        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
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
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]


        #Add model type by crownpku， bilstm or idcnn
        self.model_type = config['model_type']
        #parameters for idcnn
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0



        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)

        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)

        else:
            raise KeyError

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    #IDCNN layer
    def IDCNN_layer(self, model_inputs,
                    name=None):
        """
        :param model_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)
        reuse = False
        if not self.is_train:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            # print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iob_to_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iob_to_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)