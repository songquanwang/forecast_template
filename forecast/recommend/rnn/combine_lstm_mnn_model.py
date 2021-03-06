# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_base_model import TFBaseModel
from utils import create_variable, LSTM
from combine_data_reader import DataReader
from data_frame import DataFrame
import os
import pandas as pd
import numpy as np
from time import gmtime, strftime

from tensorflow.contrib.metrics import f1_score

mnn_feat_columns1 = ['sid', 'max_dist', 'min_dist', 'mean_dist', 'std_dist', 'max_price', 'min_price',
                     'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                     'mode_feas_1', 'mode_feas_2', 'mode_feas_3', 'mode_feas_4', 'mode_feas_5', 'mode_feas_6',
                     'mode_feas_7', 'mode_feas_8', 'mode_feas_9', 'mode_feas_10',
                     'mode_feas_11', 'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                     'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                     'svd_mode_9']

# profile svd特征
bkp_columns = ['svd_fea_0', 'svd_fea_1', 'svd_fea_2', 'svd_fea_3',
               'svd_fea_4', 'svd_fea_5', 'svd_fea_6', 'svd_fea_7', 'svd_fea_8',
               'svd_fea_9', 'svd_fea_10', 'svd_fea_11', 'svd_fea_12', 'svd_fea_13',
               'svd_fea_14', 'svd_fea_15', 'svd_fea_16', 'svd_fea_17', 'svd_fea_18',
               'svd_fea_19']
mnn_feat_columns2 = ['first_mode', 'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode',
                     'max_eta_mode', 'min_eta_mode']


class Lstm_Model(TFBaseModel):

    def __init__(self, num_encode_steps=7, **kwargs):
        self.num_encode_steps = num_encode_steps
        self.profile_data_dim = 66
        self.batch_size = 128
        self.n_hidden_units = 64
        self.num_class = 12
        self.mnn_feat_num1 = 53
        self.mnn_feat_num2 = 7
        super(Lstm_Model, self).__init__(**kwargs)

    def create_features(self):
        """
        base_columns = ['sid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'click_mode',
                'distance_list', 'eta_list', 'price_list', 'transport_mode_list']
        :return:
        """
        # batch
        self.sid = tf.placeholder(tf.int32, [None], name='ph_sid')
        # label
        self.click_mode = tf.placeholder(tf.int32, [None], name='ph_click_mode')
        # batch 每条数据实际长度
        self.encode_len = tf.placeholder(tf.int32, [None], name='ph_encode_len')
        # 用户特征，非时序
        self.profile_data = tf.placeholder(tf.float32, [None, self.profile_data_dim], name='ph_profile_data')
        # 时序特征
        self.distance_encode = tf.placeholder(tf.float32, [None, self.num_encode_steps], name='ph_distance_encode')
        self.eta_encode = tf.placeholder(tf.float32, [None, self.num_encode_steps], name='ph_eta_encode')
        self.price_encode = tf.placeholder(tf.float32, [None, self.num_encode_steps], name='ph_price_encode')
        self.mode_encode = tf.placeholder(tf.int32, [None, self.num_encode_steps], name='ph_mode_encode')
        # 非时序特征
        self.weekday = tf.placeholder(tf.int32, [None], name='ph_weekday')
        self.hour = tf.placeholder(tf.int32, [None], name='ph_hour')
        self.o1 = tf.placeholder(tf.float32, [None], name='ph_o1')
        self.o2 = tf.placeholder(tf.float32, [None], name='ph_o2')
        self.d1 = tf.placeholder(tf.float32, [None], name='ph_d1')
        self.d2 = tf.placeholder(tf.float32, [None], name='ph_d2')
        self.d2 = tf.placeholder(tf.float32, [None], name='ph_d2')
        self.euc_dis = tf.placeholder(tf.float32, [None], name='ph_euc_dis')
        # 多层神经网络
        self.mnn_feat1 = tf.placeholder(tf.float32, [None, self.mnn_feat_num1], name='ph_mnn_feat1')
        self.mnn_feat2 = tf.placeholder(tf.float32, [None, self.mnn_feat_num2], name='ph_mnn_feat2')
        # mask
        self.plan_mask = tf.placeholder(tf.float32, [None, self.num_class], name='ph_plan_mask')
        # dropout
        self.keep_prob = tf.placeholder(tf.float32, name='ph_keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='ph_is_training')

    def create_lstm_net(self):
        """
        生成lstm网络
        :return:
        """
        self.encode_features = tf.concat([
            tf.expand_dims(self.distance_encode, 2),
            tf.expand_dims(self.eta_encode, 2),
            tf.expand_dims(self.price_encode, 2),
            tf.one_hot(self.mode_encode, 12),
            # tf.tile(tf.reshape(self.o1, shape=(tf.shape(self.o1)[0], 1, 1)), (1, self.num_encode_steps, 1)),
            # tf.tile(tf.reshape(self.o2, shape=(tf.shape(self.o2)[0], 1, 1)), (1, self.num_encode_steps, 1)),
            # tf.tile(tf.reshape(self.d1, shape=(tf.shape(self.d1)[0], 1, 1)), (1, self.num_encode_steps, 1)),
            # tf.tile(tf.reshape(self.d2, shape=(tf.shape(self.d2)[0], 1, 1)), (1, self.num_encode_steps, 1)),
            # tf.tile(tf.reshape(self.euc_dis, shape=(tf.shape(self.euc_dis)[0], 1, 1)), (1, self.num_encode_steps, 1))
        ], axis=2)
        # encoder_features = tf.concat([self.encode_features, embedding_featrues, embedding_profile_feature], axis=2)
        outputs, final_state = LSTM(self.encode_features, self.encode_len, self.batch_size, self.n_hidden_units)
        # 添加注意力

        return outputs, final_state

    def create_mnn_net(self):
        """
        生成mnn网络
        :return:
        """
        embedding_dic = self.create_embedding()
        # 对时间属性embedding 21+5
        embedding_featrues = self.encoder_embedding(embedding_dic)
        # 66个用户特征降维24
        embedding_profile_feature = tf.layers.dense(self.profile_data, 10, activation=tf.nn.relu)
        # 60个gbdt特征 降维 35
        mnn_input_feature = tf.layers.dense(self.mnn_feat1, 28, activation=tf.nn.relu)

        # 合并输入特征 64 维+5
        mnn_feature = tf.concat([embedding_featrues, embedding_profile_feature, mnn_input_feature,
                                 tf.expand_dims(self.o1, 1),
                                 tf.expand_dims(self.o2, 1),
                                 tf.expand_dims(self.d1, 1),
                                 tf.expand_dims(self.d2, 1),
                                 tf.expand_dims(self.euc_dis, 1),
                                 ], axis=1)

        # 加5层全连接网络 第一层输出128
        o1 = tf.layers.dense(mnn_feature, 64, activation=tf.nn.relu)
        ob1 = tf.layers.batch_normalization(o1, training=self.is_training)
        od1 = tf.nn.dropout(ob1, self.keep_prob)
        # 第二层输入128
        o2 = tf.layers.dense(od1, 64, activation=tf.nn.relu)
        ob2 = tf.layers.batch_normalization(o2, training=self.is_training)
        od2 = tf.nn.dropout(ob2, self.keep_prob)
        # 第三层输入skip
        o3 = tf.layers.dense(od1 + od2, 64, activation=tf.nn.relu)
        ob3 = tf.layers.batch_normalization(o3, training=self.is_training)
        od3 = tf.nn.dropout(ob3, self.keep_prob)
        # 第四层输入skip
        o4 = tf.layers.dense(od2 + od3, 64, activation=tf.nn.relu)
        ob4 = tf.layers.batch_normalization(o4, training=self.is_training)
        od4 = tf.nn.dropout(ob4, self.keep_prob)
        return od4

    def create_embedding(self):
        """
         创建embedding variable
        :return:
        """
        # emb_profile_encode = create_variable('emb_profile_encode', shape=(66, 10))
        # 'first_mode', 'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode','max_eta_mode', 'min_eta_mode'
        emb_first_mode = create_variable('emb_first_mode', shape=(12, 3))
        emb_max_dist_mode = create_variable('emb_max_dist_mode', shape=(13, 3))
        emb_min_dist_mode = create_variable('emb_min_dist_mode', shape=(13, 3))
        emb_max_price_mode = create_variable('emb_max_price_mode', shape=(13, 3))
        emb_min_price_mode = create_variable('emb_min_price_mode', shape=(13, 3))
        emb_max_eta_mode = create_variable('emb_max_eta_mode', shape=(13, 3))
        emb_min_eta_mode = create_variable('emb_min_eta_mode', shape=(13, 3))
        # date
        emb_weekday = create_variable('emb_weekday', shape=(7, 2))
        emb_hour = create_variable('emb_hour', shape=(24, 3))
        return {
            'emb_first_mode': emb_first_mode,
            'emb_max_dist_mode': emb_max_dist_mode,
            'emb_min_dist_mode': emb_min_dist_mode,
            'emb_max_price_mode': emb_max_price_mode,
            'emb_min_price_mode': emb_min_price_mode,
            'emb_max_eta_mode': emb_max_eta_mode,
            'emb_min_eta_mode': emb_min_eta_mode,
            'emb_weekday': emb_weekday,
            'emb_hour': emb_hour}

    def encoder_embedding(self, embedding_dic):
        """
        encoding embeding
        :param embedding_dic:
        :return:
        """
        # eb_profile_encode = tf.nn.embedding_lookup(embedding_dic['emb_profile_encode'], tf.cast(self.profile_features, tf.int32))
        # 'first_mode', 'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode','max_eta_mode', 'min_eta_mode'
        emb_first_mode = tf.nn.embedding_lookup(embedding_dic['emb_first_mode'], tf.cast(self.mnn_feat2[:, 0], tf.int32))
        emb_max_dist_mode = tf.nn.embedding_lookup(embedding_dic['emb_max_dist_mode'], tf.cast(self.mnn_feat2[:, 1], tf.int32))
        emb_min_dist_mode = tf.nn.embedding_lookup(embedding_dic['emb_min_dist_mode'], tf.cast(self.mnn_feat2[:, 2], tf.int32))
        emb_max_price_mode = tf.nn.embedding_lookup(embedding_dic['emb_max_price_mode'], tf.cast(self.mnn_feat2[:, 3], tf.int32))
        emb_min_price_mode = tf.nn.embedding_lookup(embedding_dic['emb_min_price_mode'], tf.cast(self.mnn_feat2[:, 4], tf.int32))
        emb_max_eta_mode = tf.nn.embedding_lookup(embedding_dic['emb_max_eta_mode'], tf.cast(self.mnn_feat2[:, 5], tf.int32))
        emb_min_eta_mode = tf.nn.embedding_lookup(embedding_dic['emb_min_eta_mode'], tf.cast(self.mnn_feat2[:, 6], tf.int32))
        # date
        eb_weekday = tf.nn.embedding_lookup(embedding_dic['emb_weekday'], tf.cast(self.weekday, tf.int32))
        eb_hour = tf.nn.embedding_lookup(embedding_dic['emb_hour'], tf.cast(self.hour, tf.int32))
        # 26
        encoder_embedding_features = tf.concat([emb_first_mode, emb_max_dist_mode, emb_min_dist_mode, emb_max_price_mode,
                                                emb_min_price_mode, emb_max_eta_mode, emb_min_eta_mode,
                                                eb_weekday, eb_hour], axis=1)
        return encoder_embedding_features

    def calculate_loss(self):
        """
        训练的入口，计算损失函数
        :return:
        """
        # 销量的对数
        self.create_features()
        nnn_outputs = self.create_mnn_net()
        lstm_outputs, lstm_final_state = self.create_lstm_net()
        c, h = lstm_final_state
        # 添加注意力

        cat_outputs = tf.concat([nnn_outputs, h], axis=1)
        # 加两层全连接网络
        output_d1 = tf.layers.dense(cat_outputs, self.n_hidden_units, activation=tf.nn.relu)
        output_d2 = tf.layers.dense(output_d1, self.num_class, activation=None)
        # 对预测结果进行掩码,使预测结果在推荐列表中
        mask_output = output_d2 * self.plan_mask
        # 预测结果 0-1之间
        self.preds = tf.argmax(tf.nn.softmax(mask_output), axis=-1)
        self.labels = tf.one_hot(self.click_mode, 12)
        # 添加权重
        weight_list = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        weight_tensor = tf.convert_to_tensor(weight_list)
        weight_labels = self.labels * weight_tensor
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mask_output, labels=weight_labels), axis=-1)
        # self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=mask_output, labels=weight_labels), axis=-1)
        # (batch,64)
        self.score = tf.reduce_mean(tf.cast(tf.equal(self.click_mode, tf.cast(self.preds, tf.int32)), tf.float32))
        # self.score = f1_score()
        self.prediction_tensors = {
            'labels': self.click_mode,
            'preds': self.preds,
            'sid': self.sid,
        }

        return self.loss


def process_label_imbalance(raw_df):
    """
    处理label不均衡问题
    :param raw_df:
    :return:
    """
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 3]] * 3, ignore_index=True)
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 4]] * 4, ignore_index=True)
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 8]] * 10, ignore_index=True)
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 11]] * 4, ignore_index=True)
    return raw_df


def get_feat_data():
    """
    获取gbdt 所有特征
    :return:
    """

    mnn_feature_df = pd.read_csv('./data/recommend/features_new.csv')
    return mnn_feature_df[mnn_feat_columns1 + mnn_feat_columns2]


def get_raw_data():
    """
    获取原始数据
    :return:
    """
    all_df = pd.read_csv('./data/recommend/all_features_sta_2000.csv')
    train_df = all_df[all_df['click_mode'] != -1].head(1000)
    test_df = all_df[all_df['click_mode'] == -1].head(1000)
    all_df = pd.concat([train_df, test_df], axis=0)
    # 把-1改成12
    all_df.loc[all_df['max_dist_mode'] == -1, mnn_feat_columns2[1:]] = 12
    processed_df = process_label_imbalance(all_df)
    processed_df['distance_list'] = processed_df['distance_list'].apply(lambda x: eval(x))
    processed_df['eta_list'] = processed_df['eta_list'].apply(lambda x: eval(x))
    processed_df['price_list'] = processed_df['price_list'].apply(lambda x: eval(x))
    processed_df['transport_mode_list'] = processed_df['transport_mode_list'].apply(lambda x: eval(x))
    processed_df['plan_mask'] = processed_df['plan_mask'].apply(lambda x: eval(x))
    # 分类训练集、测试集
    all_train_data = processed_df[processed_df['click_mode'] != -1]
    all_test_data = processed_df[processed_df['click_mode'] == -1]
    return all_train_data, all_test_data


if __name__ == '__main__':
    """
        整个程序入口
    """
    base_columns = ['sid', 'pid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'euc_dis', 'click_mode',
                    'distance_list', 'eta_list', 'price_list', 'transport_mode_list', 'plan_len']
    profile_columns = ['pid'] + ['p{}'.format(i) for i in range(66)]

    all_train_data, all_test_data = get_raw_data()
    columns = base_columns + ['plan_mask', 'mnn_feat1', 'mnn_feat2', 'profile_data']
    # 构造训练data
    train_data = [all_train_data[c].values for c in base_columns]
    train_data.append(np.stack(all_train_data['plan_mask'], axis=0))
    train_data.append(all_train_data[mnn_feat_columns1[1:]].values)
    train_data.append(all_train_data[mnn_feat_columns2].values)
    train_data.append(all_train_data[profile_columns[1:]].values)
    # 构造测试data
    test_data = [all_test_data[c].values for c in base_columns]
    test_data.append(np.stack(all_test_data['plan_mask'], axis=0))
    test_data.append(all_test_data[mnn_feat_columns1[1:]].values)
    test_data.append(all_test_data[mnn_feat_columns2].values)
    test_data.append(all_test_data[profile_columns[1:]].values)

    # 构造DataFrame
    # train_df = DataFrame(data=train_data, columns=columns)
    # test_df = DataFrame(data=test_data, columns=columns)

    ################
    base_dir = './data/recommend1/'
    dr = DataReader(train_data, test_data, columns)
    nn = Lstm_Model(
        #
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        #
        learning_rate=.001,
        batch_size=128,
        num_training_steps=1,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        #
        regularization_constant=0.0,
        keep_prob=0.5,
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
    predict_df = nn.predict()
    # 保存预测结果
    submit_df = predict_df[['sid', 'preds']]
    print('***********************************************************')
    print(submit_df.columns)
    submit_df.columns = ['sid', 'recommend_mode']
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    submit_df.to_csv('{}/{}_result_{}.csv'.format(base_dir, 'lstm', now_time), index=False)
