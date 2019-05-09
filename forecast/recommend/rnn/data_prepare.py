# -*- coding: utf-8 -*-
"""
  @Author: songquanwang
  @Date: 2019-04-17 19:32:26
  @Last Modified time: 2019-04-17 19:32:26
"""

import numpy as np
import pandas as pd

class_num = 12
mnn_feat_columns = ['sid', 'first_mode', 'max_dist', 'min_dist', 'mean_dist', 'std_dist', 'max_price', 'min_price',
                    'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                    'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode',
                    'max_eta_mode', 'min_eta_mode', 'mode_feas_1', 'mode_feas_2',
                    'mode_feas_3', 'mode_feas_4', 'mode_feas_5', 'mode_feas_6',
                    'mode_feas_7', 'mode_feas_8', 'mode_feas_9', 'mode_feas_10',
                    'mode_feas_11', 'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                    'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                    'svd_mode_9', 'svd_fea_0', 'svd_fea_1', 'svd_fea_2', 'svd_fea_3',
                    'svd_fea_4', 'svd_fea_5', 'svd_fea_6', 'svd_fea_7', 'svd_fea_8',
                    'svd_fea_9', 'svd_fea_10', 'svd_fea_11', 'svd_fea_12', 'svd_fea_13',
                    'svd_fea_14', 'svd_fea_15', 'svd_fea_16', 'svd_fea_17', 'svd_fea_18',
                    'svd_fea_19']


def read_profile_data():
    """
    添加了一个全0的行
    :return:
    """
    profile_data = pd.read_csv('../data/data_set_phase1/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data


def merge_raw_data():
    """
    req_time plan_time click_time 基本相等
    :return:
    """
    # 500000 sid 唯一 ;163979 pid为null;8946 个sid没有tr_plans;46664 没有tr_click
    # sid pid req_time o d
    tr_queries = pd.read_csv('../data/data_set_phase1/train_queries.csv')
    # 94358 sid 唯一;30878 pid 为null ;1787个sid没有te_plans
    te_queries = pd.read_csv('../data/data_set_phase1/test_queries.csv')
    # 491054 sid 唯一; 无缺失值
    # sid plan_time plans([1, 2, 3, 4, 5, 6, 7]) 有1-7个plan
    # plan 格式{distance eta price transport_mode:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
    tr_plans = pd.read_csv('../data/data_set_phase1/train_plans.csv')
    # 92571 sid 唯一; 无缺失值
    te_plans = pd.read_csv('../data/data_set_phase1/test_plans.csv')
    # 453336 sid 唯一; 无缺失值
    # sid click_time click_mode：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    tr_click = pd.read_csv('../data/data_set_phase1/train_clicks.csv')
    # 训练数据
    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    # tr_data = tr_data.drop(['click_time'], axis=1)
    # 左连接不上的label置0
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)
    # 测试数据
    te_data = te_queries.merge(te_plans, on='sid', how='left')
    # label置-1
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    # data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    # 填充plans 为空数组
    data.loc[data['plans'].isnull(), 'plans'] = '[]'
    data['pid'] = data['pid'].fillna(-1)
    return data


def add_od_feas(data):
    """
    经度、维度分开
    :param data:
    :return:
    """
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    return data


def add_time_feas(data):
    """
    查询时间特征
    :param data:
    :return:
    """
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    return data


def add_plan_columns(data):
    """
    添加plans 特征
    :param data:
    :return:
    """

    #  'distance', 'eta', 'price', 'transport_mode'
    def process_plan(pl, key):
        if key == 'price':
            values = [p[key] if p[key] != '' else 0 for p in pl]
        else:
            values = [p[key] for p in pl]
        return values

    data['plans'] = data['plans'].apply(lambda x: eval(x))
    data['distance_list'] = data['plans'].apply(lambda x: process_plan(x, 'distance'))
    data['eta_list'] = data['plans'].apply(lambda x: process_plan(x, 'eta'))
    data['price_list'] = data['plans'].apply(lambda x: process_plan(x, 'price'))
    data['transport_mode_list'] = data['plans'].apply(lambda x: process_plan(x, 'transport_mode'))
    data['plan_len'] = data['plans'].apply(lambda x: len(x))

    # 添加plan mask
    def add_mask(x):
        mask = np.zeros(class_num)
        mask[x] = 1
        return list(mask)

    data['plan_mask'] = data['transport_mode_list'].apply(lambda x: add_mask(x))
    return data


def norm_value1(value_list, me, s, mi):
    """
    使用平均值、方差标准化数据
    :param value_list:
    :param m:
    :param s:
    :return:
    """
    z = np.tile(mi - 1, len(value_list))
    value_arr = np.array(value_list)
    morm_arr = np.where(value_arr > 0, (value_arr - me) / s, z)
    return list(morm_arr)


def norm_value(value_list, me, s, mi):
    """
    使用平均值、方差标准化数据
    :param value_list:
    :param m:
    :param s:
    :return:
    """
    value_arr = np.array(value_list)
    morm_arr = (value_arr - me) / s
    return list(morm_arr)


def norm_data(raw_df):
    """
    标准化数据
    :param raw_df:
    :return:
    """
    distance_cat = np.concatenate(raw_df['distance_list'])
    eta_cat = np.concatenate(raw_df['eta_list'])
    price_cat = np.concatenate(raw_df['price_list'])
    distance_arr = distance_cat[np.where(distance_cat > 0)]
    eta_arr = eta_cat[np.where(eta_cat > 0)]
    price_arr = price_cat[np.where(price_cat > 0)]
    distance_mean, distance_std, distance_min = np.mean(distance_arr), np.std(distance_arr), np.min(distance_arr)
    eta_mean, eta_std, eta_min = np.mean(eta_arr), np.std(eta_arr), np.min(eta_arr)
    price_mean, price_std, price_min = np.mean(price_arr), np.std(price_arr), np.min(price_arr)
    raw_df['distance_list'] = raw_df['distance_list'].apply(lambda x: norm_value(x, distance_mean, distance_std, distance_min))
    raw_df['eta_list'] = raw_df['eta_list'].apply(lambda x: norm_value(x, eta_mean, eta_std, eta_min))
    raw_df['price_list'] = raw_df['price_list'].apply(lambda x: norm_value(x, price_mean, price_std, price_min))
    # 经纬度标准化
    raw_df['o1'] = (raw_df['o1'] - raw_df['o1'].min()) / (raw_df['o1'].max() - raw_df['o1'].min())
    raw_df['o2'] = (raw_df['o2'] - raw_df['o2'].min()) / (raw_df['o2'].max() - raw_df['o2'].min())
    raw_df['d1'] = (raw_df['d1'] - raw_df['d1'].min()) / (raw_df['d1'].max() - raw_df['d1'].min())
    raw_df['d2'] = (raw_df['d2'] - raw_df['d2'].min()) / (raw_df['d2'].max() - raw_df['d2'].min())
    # 计算距离
    raw_df['euc_dis'] = np.sqrt(np.power((raw_df['d1'] - raw_df['o1']), 2) + np.power((raw_df['d2'] - raw_df['o2']), 2))
    return raw_df


def merge_gbdt_features():
    """
    gbdt 特征合并到lstm
    :return:
    """
    lstm_df = pd.read_csv('./data/recommend/processed_data_new.csv')
    # 获取gbdt 所有特征
    mnn_feature_df = pd.read_csv('./data/recommend/features_new.csv')[mnn_feat_columns]
    # 合并两套数据
    all_df = pd.merge(lstm_df, mnn_feature_df, on=['sid'], how='inner')
    all_df.to_csv('./data/recommend/all_features.csv', index=False)


if __name__ == '__main__':
    # base_columns = ['sid', 'pid', 'weekday', 'hour', 'o1', 'o2', 'd1', 'd2', 'click_mode',
    #                 'distance_list', 'eta_list', 'price_list', 'transport_mode_list', 'plan_mask', 'plan_len', 'euc_dis']
    # profile_columns = ['pid'] + ['p{}'.format(i) for i in range(66)]
    # data = merge_raw_data()
    # data = add_od_feas(data)
    # data = add_plan_columns(data)
    # data = add_time_feas(data)
    # profile_df = read_profile_data()
    # # data['pid'] = data['pid'].fillna(-1)
    # data = data.merge(profile_df, on='pid', how='left')
    # data[base_columns + profile_columns[1:]].to_csv('./data/processed_data_new.csv', index=False)
    # data[profile_columns].to_csv('profile_data.csv', index=False)
    merge_gbdt_features()
