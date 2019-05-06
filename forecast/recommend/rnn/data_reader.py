import os

import numpy as np

from data_frame import DataFrame
import tensorflow as tf
import pandas as pd


class DataReader(object):

    def __init__(self, data, columns):
        # 把原始数据构造成DataFrame 145063
        self.test_df = DataFrame(data=data, columns=columns)
        # 137809  7254 横向切分
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=128,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=10000,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        """
        从原始数据源dataframe中提取数据，加工成 tf placeholder所需要的内容
        注意：可以像web traffic 那样placeholder 与原始数据挂钩，然后通过tensor变换
        生成神经网络需要的数据格式
        :param batch_size:
        :param df:
        :param shuffle:
        :param num_epochs:
        :param is_test:
        :return:
        """
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )
        data_col = 'test_data' if is_test else 'data'
        is_nan_col = 'test_is_nan' if is_test else 'is_nan'
        for batch in batch_gen:
            # 预测64天
            num_decode_steps = 64
            # 数据包含的天数
            full_seq_len = batch[data_col].shape[1]
            # 804-64 如果test:804 train:740
            max_encode_length = full_seq_len - num_decode_steps if not is_test else full_seq_len
            # batch_size*804
            x_encode = np.zeros([len(batch), max_encode_length])
            # batch_size*64
            y_decode = np.zeros([len(batch), num_decode_steps])
            # batch_size*804
            is_nan_encode = np.zeros([len(batch), max_encode_length])
            # batch_size*64
            is_nan_decode = np.zeros([len(batch), num_decode_steps])
            # [0,0,....,0] batch_size个0
            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])

            for i, (seq, nan_seq) in enumerate(zip(batch[data_col], batch[is_nan_col])):
                # [375-740]  365 的区间取随机长度
                rand_len = np.random.randint(max_encode_length - 365 + 1, max_encode_length + 1)
                # 训练取随机长度；test取最大长度804
                x_encode_len = max_encode_length if is_test else rand_len
                # 从0 开始取随机长度？ 是不是可以改成 随机起点 终点
                x_encode[i, :x_encode_len] = seq[:x_encode_len]
                is_nan_encode[i, :x_encode_len] = nan_seq[:x_encode_len]
                # 记录随机长度
                encode_len[i] = x_encode_len
                decode_len[i] = num_decode_steps
                if not is_test:
                    # decode 紧邻 encode
                    y_decode[i, :] = seq[x_encode_len: x_encode_len + num_decode_steps]
                    is_nan_decode[i, :] = nan_seq[x_encode_len: x_encode_len + num_decode_steps]
            # place_holder 包含：'page_id','project','access','agent' 不包含：'test_data','test_is_nan','data', 'is_nan',
            batch['x_encode'] = x_encode
            batch['encode_len'] = encode_len
            batch['y_decode'] = y_decode
            batch['decode_len'] = decode_len
            batch['is_nan_encode'] = is_nan_encode
            batch['is_nan_decode'] = is_nan_decode

            yield batch

        def get_input_sequences(self):
            """
            base_columns = ['sid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'click_mode',
                    'distance_list', 'eta_list', 'price_list', 'transport_mode_list']
            :return:
            """
            # batch,encode_steps
            self.sid = tf.placeholder(tf.int32, [None, None])
            # batch 每条数据实际长度
            self.encode_len = tf.placeholder(tf.int32, [None])
            # batch,decode_steps
            self.y_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])
            # batch
            self.decode_len = tf.placeholder(tf.int32, [None])
            # batch ,decode_steps
            self.is_nan_encode = tf.placeholder(tf.float32, [None, None])
            # batch,decode_steps
            self.is_nan_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])
            # 其他特征
            self.page_id = tf.placeholder(tf.int32, [None])
            self.project = tf.placeholder(tf.int32, [None])
            self.access = tf.placeholder(tf.int32, [None])
            self.agent = tf.placeholder(tf.int32, [None])
            # dropout
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            # 每行求对数- 再求平均值
            self.log_x_encode_mean = sequence_mean(tf.log(self.x_encode + 1), self.encode_len)
            self.log_x_encode = self.transform(self.x_encode)
            # 销量的对数
            self.x = tf.expand_dims(self.log_x_encode, 2)
            #  batch ts feature(1+1+1+9+3+2=17）
            self.encode_features = tf.concat([
                #  nan销量0 1 掩码
                tf.expand_dims(self.is_nan_encode, 2),
                #  0销量 0 1 掩码
                tf.expand_dims(tf.cast(tf.equal(self.x_encode, 0.0), tf.float32), 2),
                tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.x_encode)[1], 1)),
                tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, tf.shape(self.x_encode)[1], 1)),
                tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, tf.shape(self.x_encode)[1], 1)),
                tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, tf.shape(self.x_encode)[1], 1)),
            ], axis=2)
            # (batch ,64) 位置信息(0,1,2,...,num_decode_steps)
            decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps), 0), (tf.shape(self.y_decode)[0], 1))
            # (batch,64,64),features(64,1+9+3+2) 没有is_nan_encode,x_encode
            self.decode_features = tf.concat([
                # 把每一步独热编码
                tf.one_hot(decode_idx, self.num_decode_steps),
                tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, self.num_decode_steps, 1)),
                tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, self.num_decode_steps, 1)),
                tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, self.num_decode_steps, 1)),
                tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, self.num_decode_steps, 1)),
            ], axis=2)

            return self.x


if __name__ == '__main__':
    base_df = pd.read_csv('./data/base_data.csv')
    profile_df = pd.read_csv('./data/profile_data.csv')
    base_columns = ['sid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'click_mode',
                    'distance_list', 'eta_list', 'price_list', 'transport_mode_list']
    profile_columns = ['p{}'.format(i) for i in range(66)]
    columns = base_columns + ['profile_data']

    base_data = [base_df[c].values for c in base_columns]
    base_data.append(profile_df[profile_columns].values)
    test_df = DataFrame(data=base_data, columns=columns)
