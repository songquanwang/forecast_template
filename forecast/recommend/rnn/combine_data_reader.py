# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from data_frame import DataFrame


class DataReader(object):

    def __init__(self, train_data, test_data, columns):
        #
        self.train_all_df = DataFrame(data=train_data, columns=columns)
        self.test_df = DataFrame(data=test_data, columns=columns)
        # 137809  7254 横向切分
        self.train_df, self.val_df = self.train_all_df.train_test_split(train_size=0.20)

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
            num_epochs=1,
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
        max_plan_length = 7
        for batch in batch_gen:
            # 保存推荐计划个数
            encode_len = np.zeros([len(batch)])
            # 保存推荐计划 distance eta price mode
            distance_encode = np.zeros([len(batch), max_plan_length])
            eta_encode = np.zeros([len(batch), max_plan_length])
            price_encode = np.zeros([len(batch), max_plan_length])
            mode_encode = np.zeros([len(batch), max_plan_length])
            # 为计划赋值，不足7 后面补充0
            for i in range(len(batch)):
                plan_len = int(batch[i]['plan_len'])
                # encode_len[i] = plan_len
                distance_encode[i, 0:plan_len] = batch[i].distance_list[0:plan_len]
                eta_encode[i, 0:plan_len] = batch[i].eta_list[0:plan_len]
                price_encode[i, 0:plan_len] = batch[i].price_list[0:plan_len]
                mode_encode[i, 0:plan_len] = batch[i].transport_mode_list[0:plan_len]

            # 把特征放到batch中
            batch['distance_encode'] = distance_encode
            batch['eta_encode'] = eta_encode
            batch['price_encode'] = price_encode
            batch['mode_encode'] = mode_encode
            batch['encode_len'] = encode_len

            yield batch


if __name__ == '__main__':
    pass
