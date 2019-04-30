# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-04-17 19:32:26 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-04-17 19:32:26 
"""

import json
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


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
    tr_data = tr_queries.merge(tr_click, on='sid', how='inner')
    tr_data = tr_data.merge(tr_plans, on='sid', how='inner')
    tr_data = tr_data.drop(['click_time'], axis=1)
    # 左连接不上的label置0
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)
    # 测试数据
    te_data = te_queries.merge(te_plans, on='sid', how='inner')
    # label置-1
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    # data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def get_plan_df(data):
    """
    获取
    :return:
    """
    data.loc[data['click_mode'] != -1, 'is_train'] = 1
    data.loc[data['click_mode'] == -1, 'is_train'] = 0
    data['plans'] = data['plans'].apply(lambda line: eval(line))
    lens = [len(item) for item in data['plans']]
    # plan_time distance eta price transport_mode
    sid_list = np.repeat(data['sid'].values, lens)
    is_train_list = np.repeat(data['is_train'].values, lens)
    plan_time_list = np.repeat(data['plan_time'].values, lens)
    plans = np.concatenate(data['plans'].values)
    df_data = []
    for s, it, t, p in zip(sid_list, is_train_list, plan_time_list, plans):
        p['sid'] = s
        p['is_train'] = it
        p['plan_time'] = t
        df_data.append(p)
    # 生成新的plans_df
    plans_df = pd.DataFrame(df_data)
    plans_df = plans_df[['sid', 'is_train', 'plan_time', 'distance', 'eta', 'price', 'transport_mode']]
    plans_df['plan_time'] = pd.to_datetime(plans_df['plan_time'])
    plans_df['price'] = plans_df['price'].replace(r'', np.NaN)
    ###############
    plans_df['dj'] = plans_df['price'] * 100 / plans_df['distance']

    def convert_time(d, m):
        return (d.hour * 60 + d.minute) // m

    # 3 5 6 价格填充为0
    plans_df.loc[plans_df['transport_mode'].isin([3, 5, 6]), 'price'] = 0
    plans_df['time_num30'] = plans_df['plan_time'].apply(lambda x: convert_time(x, 30))
    plans_df['dj'] = plans_df['price'] * 100 / plans_df['distance']
    plans_df['mdj'] = plans_df.groupby(['transport_mode', 'time_num30'])['dj'].transform(lambda x: np.nanmedian(x))
    # 处理4类型的distance
    plans_df['dj1'] = plans_df['dj']
    plans_df['distance1'] = plans_df['distance']
    plans_df.loc[(plans_df['transport_mode'] == 4) & (plans_df['distance'] < 100), 'dj1'] = plans_df.loc[
        (plans_df['transport_mode'] == 4) & (plans_df['distance'] < 100), 'mdj']
    df1 = plans_df.loc[(plans_df['transport_mode'] == 4) & (plans_df['distance'] < 100)]
    plans_df.loc[(plans_df['transport_mode'] == 4) & (plans_df['distance'] < 100), 'distance1'] = df1['price'] * 100 / \
                                                                                                  df1['dj1']
    # 填充 price dj[1, 2, 7, 9, 11]
    plans_df['price_nan'] = plans_df['price'].apply(lambda x: 1 if np.isnan(x) else 0)
    plans_df['price1'] = plans_df['price']
    plans_df.loc[plans_df['price'].isnull(), 'dj1'] = plans_df.loc[plans_df['price'].isnull(), 'mdj']
    df2 = plans_df.loc[plans_df['price'].isnull()]
    plans_df.loc[plans_df['price'].isnull(), 'price1'] = df2['dj1'] * df2['distance1'] / 100
    return plans_df


def gen_od_feas(data):
    """
    经度、维度分开
    :param data:
    :return:
    """
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    data = data.drop(['o', 'd'], axis=1)
    return data


def gen_plan_feas(plans_df):
    """
    plan字段:distance ,eta, price, transport_mode
    transport_mode: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    [1, 2, 7, 9, 11] price 存在''
    [3,5,6] price全是0
    4 8 10 price不为''
    0 未知(没推荐);-1 待预测
    词向量:
    mode_texts=['word_1 word_3 word_5','word_1 word_4 word_6'
    tfidf_enc.vocabulary_
    {'word_1': 0,'word_1 word_3': 1,'word_1 word_4': 2, 'word_3': 3,'word_3 word_5': 4, 'word_4': 5, 'word_4 word_6': 6,'word_5': 7, 'word_6': 8}
    tfidf_vec.toarray() ;词向量跟词典长度相同
    array([0.33517574, 0.47107781, 0.        , 0.47107781, 0.47107781, 0.        , 0.        , 0.47107781, 0.        ])
    array([0.33517574, 0.        , 0.47107781, 0.        , 0.        ,0.47107781, 0.47107781, 0.        , 0.47107781])

    需要处理 4 distance
    :param data:
    :return:
    """
    mode_df = pd.get_dummies(plans_df['transport_mode'], prefix='mode_feas_', drop_first=False)
    # 第一种模式
    plans_df['first_mode'] = plans_df.groupby('sid')['transport_mode'].transform(lambda x: x.values[0])

    plans_df['max_dist'] = plans_df.groupby('sid')['distance1'].transform(lambda x: np.max(x))
    plans_df['min_dist'] = plans_df.groupby('sid')['distance1'].transform(lambda x: np.min(x))
    plans_df['mean_dist'] = plans_df.groupby('sid')['distance1'].transform(lambda x: np.mean(x))
    plans_df['std_dist'] = plans_df.groupby('sid')['distance1'].transform(lambda x: np.std(x))

    plans_df['max_price'] = plans_df.groupby('sid')['price1'].transform(lambda x: np.max(x))
    plans_df['min_price'] = plans_df.groupby('sid')['price1'].transform(lambda x: np.min(x))
    plans_df['mean_price'] = plans_df.groupby('sid')['price1'].transform(lambda x: np.mean(x))
    plans_df['std_price'] = plans_df.groupby('sid')['price1'].transform(lambda x: np.std(x))

    plans_df['max_eta'] = plans_df.groupby('sid')['eta'].transform(lambda x: np.max(x))
    plans_df['min_eta'] = plans_df.groupby('sid')['eta'].transform(lambda x: np.min(x))
    plans_df['mean_eta'] = plans_df.groupby('sid')['eta'].transform(lambda x: np.mean(x))
    plans_df['std_eta'] = plans_df.groupby('sid')['eta'].transform(lambda x: np.std(x))

    plans_df['mode_texts'] = plans_df.groupby('sid')['transport_mode'].transform(
        lambda x: ' '.join(['word_{}'.format(mode) for mode in x.values]))

    # 计算距离模式
    max_dist_mode_df = plans_df.loc[plans_df['distance1'] == plans_df['max_dist'], ['sid', 'transport_mode']]
    max_dist_mode_df.columns = ['sid', 'max_dist_mode']
    max_dist_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    min_dist_mode_df = plans_df.loc[plans_df['distance1'] == plans_df['min_dist'], ['sid', 'transport_mode']]
    min_dist_mode_df.columns = ['sid', 'min_dist_mode']
    min_dist_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    max_price_mode_df = plans_df.loc[plans_df['price1'] == plans_df['max_price'], ['sid', 'transport_mode']]
    max_price_mode_df.columns = ['sid', 'max_price_mode']
    max_price_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    min_price_mode_df = plans_df.loc[plans_df['price1'] == plans_df['min_price'], ['sid', 'transport_mode']]
    min_price_mode_df.columns = ['sid', 'min_price_mode']
    min_price_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    max_eta_mode_df = plans_df.loc[plans_df['eta'] == plans_df['max_eta'], ['sid', 'transport_mode']]
    max_eta_mode_df.columns = ['sid', 'max_eta_mode']
    max_eta_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    min_eta_mode_df = plans_df.loc[plans_df['eta'] == plans_df['min_eta'], ['sid', 'transport_mode']]
    min_eta_mode_df.columns = ['sid', 'min_eta_mode']
    min_eta_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)

    # 合并特征

    plans_df = pd.merge(plans_df, max_dist_mode_df, on=['sid'], how='inner')
    plans_df = pd.merge(plans_df, min_dist_mode_df, on=['sid'], how='inner')
    plans_df = pd.merge(plans_df, max_price_mode_df, on=['sid'], how='inner')
    plans_df = pd.merge(plans_df, min_price_mode_df, on=['sid'], how='inner')
    plans_df = pd.merge(plans_df, max_eta_mode_df, on=['sid'], how='inner')
    plans_df = pd.merge(plans_df, min_eta_mode_df, on=['sid'], how='inner')

    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(plans_df['mode_texts'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    plans_df = pd.concat([plans_df, mode_df, mode_svd], axis=1)
    return plans_df


def get_plan_feas():
    plan_featurns = pd.read_csv('../data/data_set_phase1/plan_features.csv')
    return plan_featurns


def gen_profile_feas(data):
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1)
    data = data.merge(svd_feas, on='pid', how='left')
    return data


def gen_time_feas(data):
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    data = data.drop(['req_time'], axis=1)
    return data


def split_train_test(data):
    train_data = data[data['click_mode'] != -1]
    test_data = data[data['click_mode'] == -1]
    submit = test_data[['sid']].copy()
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, submit


def gen_train_test_feas_data():
    """
    :return:
    """
    feature_columns = ['sid', 'pid', 'click_mode', 'o1', 'o2', 'd1', 'd2',
                       'eta', 'transport_mode', 'dj1', 'distance1', 'price1', 'first_mode', 'max_dist', 'min_dist',
                       'mean_dist', 'std_dist', 'max_price', 'min_price',
                       'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                       'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode',
                       'max_eta_mode', 'min_eta_mode', 'mode_feas__1', 'mode_feas__2',
                       'mode_feas__3', 'mode_feas__4', 'mode_feas__5', 'mode_feas__6',
                       'mode_feas__7', 'mode_feas__8', 'mode_feas__9', 'mode_feas__10',
                       'mode_feas__11', 'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                       'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                       'svd_mode_9', 'svd_fea_0', 'svd_fea_1', 'svd_fea_2', 'svd_fea_3',
                       'svd_fea_4', 'svd_fea_5', 'svd_fea_6', 'svd_fea_7', 'svd_fea_8',
                       'svd_fea_9', 'svd_fea_10', 'svd_fea_11', 'svd_fea_12', 'svd_fea_13',
                       'svd_fea_14', 'svd_fea_15', 'svd_fea_16', 'svd_fea_17', 'svd_fea_18',
                       'svd_fea_19', 'weekday', 'hour']
    data = merge_raw_data()
    data = gen_od_feas(data)
    plans_df = get_plan_df(data)
    plans_features = gen_plan_feas(plans_df)
    data = pd.merge(data, plans_features, on=['sid'], how='inner')
    data = gen_profile_feas(data)
    data = gen_time_feas(data)
    data = data[feature_columns]
    data.to_csv('./features.csv', index=False)
    train_x, train_y, test_x, submit = split_train_test(data)
    return train_x, train_y, test_x, submit


def get_train_test_feas_data():
    data = pd.read_csv('./featues.csv')
    train_x, train_y, test_x, submit = split_train_test(data)
    return train_x, train_y, test_x, submit


if __name__ == '__main__':
    get_train_test_feas_data()
