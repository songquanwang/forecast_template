# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_base_model import TFBaseModel
from utils import create_variable, LSTM
from data_reader import DataReader
from data_frame import DataFrame
import os
import pandas as pd


class Lstm_Model(TFBaseModel):

    def __init__(self, num_encode_steps=7, **kwargs):
        self.num_encode_steps = num_encode_steps
        self.profile_data_dim = 66
        self.batch_size = 128
        self.n_hidden_units = 64
        self.num_class = 12
        super(Lstm_Model, self).__init__(**kwargs)

    def create_features(self):
        """
        base_columns = ['sid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'click_mode',
                'distance_list', 'eta_list', 'price_list', 'transport_mode_list']
        :return:
        """
        # batch
        self.sid = tf.placeholder(tf.int32, [None])
        # label
        self.click_mode = tf.placeholder(tf.int32, [None])
        # batch 每条数据实际长度
        self.encode_len = tf.placeholder(tf.int32, [None])
        # 用户特征，非时序
        self.profile_encode = tf.placeholder(tf.float32, [None, self.profile_data_dim])
        # 时序特征
        self.distance_encode = tf.placeholder(tf.float32, [None, self.num_encode_steps])
        self.eta_encode = tf.placeholder(tf.float32, [None, self.num_encode_steps])
        self.price_encode = tf.placeholder(tf.float32, [None, self.num_encode_steps])
        self.mode_encode = tf.placeholder(tf.int32, [None, self.num_encode_steps])
        # 非时序特征
        self.weekday = tf.placeholder(tf.int32, [None])
        self.hour = tf.placeholder(tf.int32, [None])
        self.o1 = tf.placeholder(tf.float32, [None])
        self.o2 = tf.placeholder(tf.float32, [None])
        self.d1 = tf.placeholder(tf.float32, [None])
        self.d2 = tf.placeholder(tf.float32, [None])
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        #  batch feature 66+7+24=97 需要embedding的特征
        self.profile_features = tf.tile(tf.expand_dims(self.profile_encode, 1), (1, self.num_encode_steps, 1))
        self.weekday_features = tf.tile(tf.expand_dims(tf.one_hot(self.weekday, 7), 1), (1, self.num_encode_steps, 1))
        self.hour_features = tf.tile(tf.expand_dims(tf.one_hot(self.hour, 24), 1), (1, self.num_encode_steps, 1))
        # batch features 8
        self.encode_features = tf.concat([
            tf.expand_dims(self.distance_encode, 2),
            tf.expand_dims(self.eta_encode, 2),
            tf.expand_dims(self.price_encode, 2),
            tf.one_hot(self.mode_encode, 12),
            tf.tile(tf.expand_dims(self.o1, 1), (1, self.num_encode_steps, 1)),
            tf.tile(tf.expand_dims(self.o2, 1), (1, self.num_encode_steps, 1)),
            tf.tile(tf.expand_dims(self.d1, 1), (1, self.num_encode_steps, 1)),
            tf.tile(tf.expand_dims(self.d2, 1), (1, self.num_encode_steps, 1))
        ], axis=2)

        return self.encode_features

    def create_embedding(self):
        """
         创建embedding variable
        :return:
        """
        emb_profile_encode = create_variable('emb_profile_encode', shape=(66, 10))
        emb_weekday = create_variable('emb_weekday', shape=(7, 2))
        emb_hour = create_variable('emb_hour', shape=(24, 3))

        return {'emb_profile_encode': emb_profile_encode, 'emb_weekday': emb_weekday, 'emb_hour': emb_hour}

    def encoder_embedding(self, embedding_dic):
        """
        encoding embeding
        :param embedding_dic:
        :return:
        """
        eb_profile_encode = tf.nn.embedding_lookup(embedding_dic['emb_profile_encode'], tf.cast(self.profile_features, tf.int32))
        eb_weekday = tf.nn.embedding_lookup(embedding_dic['emb_weekday'], self.weekday_features)
        eb_hour = tf.nn.embedding_lookup(embedding_dic['emb_hour'], self.hour_features)
        encoder_embedding_features = tf.concat([eb_profile_encode, eb_weekday, eb_hour], axis=2)
        return encoder_embedding_features

    def calculate_loss(self):
        """
        训练的入口，计算损失函数
        :return:
        """
        # 销量的对数
        raw_features = self.create_features()
        embedding_dic = self.create_embedding()
        embedding_featrues = self.encoder_embedding(embedding_dic)
        encoder_features = tf.concat([raw_features, embedding_featrues], axis=2)

        outputs, final_state = LSTM(encoder_features, self.encode_len, self.batch_size, self.n_hidden_units)
        c, h = final_state
        # 加两层全连接网络
        output_d1 = tf.layers.dense(h, self.n_hidden_units, activation=tf.nn.relu)
        output_d2 = tf.layers.dense(output_d1 + h, self.num_class, activation=None)
        mask_output = output_d2 * self.mask
        # 预测结果 0-1之间
        self.preds = tf.argmax(tf.nn.softmax(output_d2))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_output, labels=self.click_mode), axis=-1, keepdims=True)
        # (batch,64)

        self.prediction_tensors = {
            'labels': self.click_mode,
            'preds': self.preds,
            'sid': self.sid,
        }

        return self.loss


if __name__ == '__main__':
    base_columns = ['sid', 'pid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'click_mode',
                    'distance_list', 'eta_list', 'price_list', 'transport_mode_list']
    profile_columns = ['pid'] + ['p{}'.format(i) for i in range(66)]
    processed_df = pd.read_csv('./data/processed_data.csv')
    # 分类训练集、测试集
    all_train_data = processed_df[processed_df['click_mode'] != -1]
    all_test_data = processed_df[processed_df['click_mode'] == -1]

    columns = base_columns + ['profile_data']
    # 构造训练data
    train_data = [all_train_data[c].values for c in base_columns]
    train_data.append(all_train_data[profile_columns].values)
    # 构造测试data
    test_data = [all_test_data[c].values for c in base_columns]
    train_data.append(all_train_data[profile_columns].values)

    # 构造DataFrame
    train_df = DataFrame(data=train_data, columns=columns)
    train_df = DataFrame(data=train_data, columns=columns)

    ################
    base_dir = './'
    dr = DataReader(train_data, test_data, columns)
    nn = LSTM(
        #
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        #
        learning_rate=.001,
        batch_size=128,
        num_training_steps=200000,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        #
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=500,
        #
        log_interval=10,
        num_validation_batches=1,
        grad_clip=20,
        # 子类构造属性
        num_encode_steps=7
    )
    nn.fit()
    nn.restore()
    nn.predict()
