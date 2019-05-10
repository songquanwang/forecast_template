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
from collections import OrderedDict
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from six.moves import reduce
import common


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
    1.tr_queries 中没有tr_plans 的8946
    2.tr_plans 中没有 tr_click 的占37718；tr_click是tr_plans的子集
    3.te_queries中没有te_plans的占1787
    结论:
    1.有tr_plans 的没有tr_click 37718的需要把click_mode=0 作为训练数据
    2.没有tr_plans的 8946 不需要作为训练数据
    3.没有te_plans的tr_queries 1787 直接预测为0

    d1=feature_df[(feature_df['click_mode'] ==0)&(feature_df['max_dist'] !=-1)]
    d2=feature_df[(feature_df['click_mode'] ==0)&(feature_df['max_dist'] ==-1)]
    d3=feature_df[(feature_df['click_mode'] ==-1)&(feature_df['max_dist'] ==-1)]
    # 同一个sid，有相同的transoport 推荐 110208
    110208
    395
    290
    1
    tr_plans[tr_plans['sid']==3190603].iloc[0].plans

    没有plans的query 直接预测为0(没点击)
    :return:
    """
    # 500000 sid 唯一 ;163979 pid为null;8946 个sid没有 tr_plans;46664 没有tr_click
    # 有plans 没有 click 37718
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
    tr_data = tr_data.drop(['click_time'], axis=1)
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
    return data


def get_plan_df_bkp(data):
    """
    获取 dj mdj -->dj1 distance1 price1
    生成
    :return:
    """
    data.loc[data['click_mode'] != -1, 'is_train'] = 1
    data.loc[data['click_mode'] == -1, 'is_train'] = 0
    data.loc[data['plans'].isnull(), 'plans'] = '[]'
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


def get_plan_df(data):
    """
    获取 dj mdj -->dj1 distance1 price1
    生成
    :return:
    """
    data.loc[data['click_mode'] != -1, 'is_train'] = 1
    data.loc[data['click_mode'] == -1, 'is_train'] = 0
    data.loc[data['plans'].isnull(), 'plans'] = '[]'
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
    # '' 替换成np.nan
    plans_df['price'] = plans_df['price'].replace(r'', np.NaN)

    ###############
    def convert_time(d, m):
        return (d.hour * 60 + d.minute) // m

    # 3 5 6 价格填充为0
    plans_df.loc[plans_df['transport_mode'].isin([3, 5, 6]), 'price'] = 0
    plans_df['time_num30'] = plans_df['plan_time'].apply(lambda x: convert_time(x, 30))
    plans_df['dj'] = plans_df['price'] * 100 / plans_df['distance']
    plans_df['mdj'] = plans_df.groupby(['transport_mode', 'time_num30'])['dj'].transform(lambda x: np.nanmedian(x))

    # 填充 price dj[1, 2, 7, 9, 11]
    # plans_df['price1'] = plans_df['price']
    plans_df.loc[plans_df['price'].isnull(), 'dj'] = plans_df.loc[plans_df['price'].isnull(), 'mdj']
    df2 = plans_df.loc[plans_df['price'].isnull()]
    plans_df.loc[plans_df['price'].isnull(), 'price'] = df2['dj'] * df2['distance'] / 100
    # sid, plan_time, distance, eta, price, transport_mode
    return plans_df[['sid', 'plan_time', 'distance', 'eta', 'price', 'transport_mode']]


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


def gen_plan_feas_bkp(plans_df):
    """
    太慢了
    plan字段:distance ,eta, price, transport_mode
    transport_mode: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    [1, 2, 7, 9, 11] price 存在''
    [3,5,6] price全是''
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
    columns_names = ['mode_feas_{}'.format(i) for i in range(12)]

    def gen_mode_code(mode_list):
        ma = np.zeros(12)
        ma[mode_list] = 1
        return ma

    data = []
    groups = plans_df.groupby('sid')

    def get_row(group):
        # transport_mode
        ls = OrderedDict()
        ls['sid'] = group['sid'].values[0]
        columns_values = list(gen_mode_code(group['transport_mode'].values))
        for key, value in zip(columns_names, columns_values):
            ls[key] = value
        ls['first_mode'] = group['transport_mode'].values[0]
        # distance
        ls['max_dist'] = group['distance'].max()
        ls['min_dist'] = group['distance'].min()
        ls['mean_dist'] = group['distance'].mean()
        ls['std_dist'] = group['distance'].std(ddof=0)
        # price
        ls['max_price'] = group['price'].max()
        ls['min_price'] = group['price'].min()
        ls['mean_price'] = group['price'].mean()
        ls['std_price'] = group['price'].std(ddof=0)
        # eta
        ls['max_eta'] = group['eta'].max()
        ls['min_eta'] = group['eta'].min()
        ls['mean_eta'] = group['eta'].mean()
        ls['std_eta'] = group['eta'].std(ddof=0)
        # mode_texts
        ls['mode_texts'] = ' '.join(['word_{}'.format(mode) for mode in group['transport_mode'].values])
        # 符合特征 df.iloc[df['D'].idxmin()].C
        ls['max_dist_mode'] = group.loc[group.index == group['distance'].idxmax()].transport_mode.values[0]
        ls['min_dist_mode'] = group.loc[group.index == group['distance'].idxmin()].transport_mode.values[0]
        ls['max_price_mode'] = group.loc[group.index == group['price'].idxmax()].transport_mode.values[0]
        ls['min_price_mode'] = group.loc[group.index == group['price'].idxmin()].transport_mode.values[0]
        ls['max_eta_mode'] = group.loc[group.index == group['eta'].idxmax()].transport_mode.values[0]
        ls['min_eta_mode'] = group.loc[group.index == group['eta'].idxmin()].transport_mode.values[0]
        return ls

    for name, group in groups:
        row = get_row(group)
        data.append(row)
    feature_df = pd.DataFrame(data)
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(feature_df['mode_texts'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    feature_df = pd.concat([feature_df, mode_svd], axis=1)
    return feature_df


def gen_plan_feas_by_plans(plans_df):
    """

    :param data:
    :return:
    """
    mode_columns_names = ['mode_feas_{}'.format(i) for i in range(12)]

    def gen_mode_code(mode_list):
        ma = np.zeros(12)
        ma[mode_list] = 1
        return ma

    mode_g = plans_df.groupby('sid')['transport_mode'].apply(gen_mode_code).reset_index()
    mode_columns = ['sid'] + mode_columns_names
    mode_data = np.concatenate(mode_g['transport_mode'].values, axis=0).reshape(len(mode_g), 12)
    sid_data = mode_g['sid'].values.reshape(len(mode_g), 1)
    mode_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)

    def get_first(x):
        return x.values[0]

    def gen_mode_texts(x):
        tl = ' '.join(['word_{}'.format(mode) for mode in x.values])
        return tl

    agg_fun = {'transport_mode': [get_first, gen_mode_texts],
               'distance': ['max', 'min', 'mean', lambda x: np.std(x)],
               'price': ['max', 'min', 'mean', lambda x: np.std(x)],
               'eta': ['max', 'min', 'mean', lambda x: np.std(x)]}
    # std ddof =1
    agg_columns = ['sid', 'first_mode', 'mode_texts', 'max_dist', 'min_dist', 'mean_dist', 'std_dist',
                   'max_price', 'min_price', 'mean_price', 'std_price',
                   'max_eta', 'min_eta', 'mean_eta', 'std_eta']
    agg_df = plans_df.groupby('sid').agg(agg_fun).reset_index()
    agg_df.columns = agg_columns
    merge_df = pd.merge(plans_df, agg_df, on=['sid'], how='inner')
    # 原来版本是 keep='last'
    max_dist_mode_df = merge_df.loc[merge_df['distance'] == merge_df['max_dist'], ['sid', 'transport_mode']]
    max_dist_mode_df.columns = ['sid', 'max_dist_mode']
    max_dist_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    min_dist_mode_df = merge_df.loc[merge_df['distance'] == merge_df['min_dist'], ['sid', 'transport_mode']]
    min_dist_mode_df.columns = ['sid', 'min_dist_mode']
    min_dist_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    max_price_mode_df = merge_df.loc[merge_df['price'] == merge_df['max_price'], ['sid', 'transport_mode']]
    max_price_mode_df.columns = ['sid', 'max_price_mode']
    max_price_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    min_price_mode_df = merge_df.loc[merge_df['price'] == merge_df['min_price'], ['sid', 'transport_mode']]
    min_price_mode_df.columns = ['sid', 'min_price_mode']
    min_price_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    max_eta_mode_df = merge_df.loc[merge_df['eta'] == merge_df['max_eta'], ['sid', 'transport_mode']]
    max_eta_mode_df.columns = ['sid', 'max_eta_mode']
    max_eta_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    min_eta_mode_df = merge_df.loc[merge_df['eta'] == merge_df['min_eta'], ['sid', 'transport_mode']]
    min_eta_mode_df.columns = ['sid', 'min_eta_mode']
    min_eta_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)

    complex_feature_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['sid'], how='inner'),
                                [max_dist_mode_df, min_dist_mode_df, max_price_mode_df, min_price_mode_df,
                                 max_eta_mode_df, min_eta_mode_df])
    plan_feature_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['sid'], how='inner'),
                             [mode_df, agg_df, complex_feature_df])

    return plan_feature_df


def gen_empty_plan_feas(data):
    """
    生成empty plans
    :param data:
    :return:
    """
    mode_columns_names = ['mode_feas_{}'.format(i) for i in range(12)]

    mode_data = np.zeros((len(data), 12))
    mode_data[:, 0] = 1
    sid_data = data['sid'].values.reshape(len(data), 1)
    mode_columns = ['sid'] + mode_columns_names
    plan_feature_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)

    plan_feature_df['first_mode'] = 0
    plan_feature_df['mode_texts'] = 'word_null'
    plan_feature_df['max_dist'] = -1
    plan_feature_df['min_dist'] = -1
    plan_feature_df['mean_dist'] = -1
    plan_feature_df['std_dist'] = -1

    plan_feature_df['max_price'] = -1
    plan_feature_df['min_price'] = -1
    plan_feature_df['mean_price'] = -1
    plan_feature_df['std_price'] = -1

    plan_feature_df['max_eta'] = -1
    plan_feature_df['min_eta'] = -1
    plan_feature_df['mean_eta'] = -1
    plan_feature_df['std_eta'] = -1
    plan_feature_df['max_dist_mode'] = -1
    plan_feature_df['min_dist_mode'] = -1
    plan_feature_df['max_price_mode'] = -1
    plan_feature_df['min_price_mode'] = -1
    plan_feature_df['max_eta_mode'] = -1
    plan_feature_df['min_eta_mode'] = -1

    return plan_feature_df


def get_plan_feas():
    plan_featurns = pd.read_csv('../data/data_set_phase1/plan_features.csv')
    return plan_featurns


def gen_plan_feas(data):
    # plans_df = get_plan_df(data)
    # tr_plans + te_plans =583625
    plans_df = pd.read_csv('../data/plans_new.csv')
    data_empty = data[~data['sid'].isin(plans_df.sid.unique())]
    plans_features = gen_plan_feas_by_plans(plans_df)
    empty_plans_features = gen_empty_plan_feas(data_empty)
    plan_feature_df = pd.concat([plans_features, empty_plans_features], axis=0).reset_index(drop=True)
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    # 添加tdidf svd
    tfidf_vec = tfidf_enc.fit_transform(plan_feature_df['mode_texts'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd_df = pd.DataFrame(mode_svd)
    mode_svd_df.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    feature_df = pd.concat([plan_feature_df, mode_svd_df], axis=1)
    return feature_df


def add_sta_feas(data):
    """
    添加统计特征
    :param data:
    :return:
    """
    dist = 'cosine'
    svd_columns = ['svd_mode_{}'.format(i) for i in range(10)]
    train_data = data[data['click_mode'] != -1]
    test_data = data[data['click_mode'] == -1]
    mode_indices_dict = common.get_sample_indices_by_relevance(train_data)
    pid_mode_indices_dict = common.get_sample_indices_by_relevance(train_data, "pid")
    stat_columns_sid = ['stat_{0}_{1}'.format(c, i) for c in ['min', 'median', 'max', 'mean', 'std'] for i in range(12)]
    cut_data = data[['sid'] + svd_columns]
    cut_data.loc[data['click_mode'] != -1, stat_columns_sid] = common.generate_dist_stats_feat(dist, train_data[svd_columns].values, train_data["sid"].values,
                                                                                               train_data[svd_columns].values, train_data["sid"].values,
                                                                                               mode_indices_dict)
    cut_data.loc[data['click_mode'] == -1, stat_columns_sid] = common.generate_dist_stats_feat(dist, train_data[svd_columns].values, train_data["sid"].values,
                                                                                               test_data[svd_columns].values, test_data["sid"].values,
                                                                                               mode_indices_dict)
    pid_stat_columns_sid = ['stat_pid_{0}'.format(c) for c in ['min', 'median', 'max', 'mean', 'std']]
    cut_data.loc[data['click_mode'] != -1, pid_stat_columns_sid] = common.generate_dist_stats_feat(dist, train_data[svd_columns.values], train_data["sid"].values,
                                                                                                   train_data[svd_columns.values], train_data["sid"].values,
                                                                                                   pid_mode_indices_dict)
    cut_data.loc[data['click_mode'] == -1, pid_stat_columns_sid] = common.generate_dist_stats_feat(dist, train_data[svd_columns.values], train_data["sid"].values,
                                                                                                   test_data[svd_columns.values], test_data["sid"].values,
                                                                                                   pid_mode_indices_dict)

    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    stat_svd = svd_enc.fit_transform(cut_data[stat_columns_sid].values)
    stat_svd_df = pd.DataFrame(stat_svd)
    stat_svd_df.columns = ['stat_fea_{}'.format(i) for i in range(20)]

    data = pd.merge(data, stat_svd_df, on=['sid'], how='inner')
    return data


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
    train_sid = train_data[['sid']].copy()
    test_sid = test_data[['sid']].copy()
    submit = test_data[['sid']].copy()
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, submit, train_sid, test_sid


def gen_train_test_feas_data():
    """
    :return:
    """
    feature_columns = ['sid', 'pid', 'click_mode', 'o1', 'o2', 'd1', 'd2',
                       'first_mode', 'max_dist', 'min_dist', 'mean_dist', 'std_dist', 'max_price', 'min_price',
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
                       'svd_fea_19', 'weekday', 'hour']
    # 添加统计特征
    # stat_columns_sid = ['stat_{0}'.format(c) for c in ['min', 'median', 'max', 'mean', 'std']]
    pid_stat_columns_sid = ['stat_pid_{0}'.format(c) for c in ['min', 'median', 'max', 'mean', 'std']]

    data = merge_raw_data()
    data = data.drop(['plans'], axis=1)
    data = gen_od_feas(data)
    plans_features = gen_plan_feas(data)
    # union没有plans的 innner=left
    data = pd.merge(data, plans_features, on=['sid'], how='left')
    data = gen_profile_feas(data)
    data = gen_time_feas(data)
    # data = add_sta_feas(data)
    data = data[feature_columns]
    # 545907 = tr_click + te_plans
    data.to_csv('../data/features_new.csv', index=False)
    train_x, train_y, test_x, submit = split_train_test(data)
    return train_x, train_y, test_x, submit


def get_train_test_feas_data_1():
    data = pd.read_csv('../data/features_new.csv')
    train_x, train_y, test_x, submit = split_train_test(data)
    return train_x, train_y, test_x, submit


def get_train_test_feas_data_3():
    data = pd.read_csv('../data/features_new.csv')
    train_data = data[data['click_mode'] != -1]
    from sklearn.model_selection import train_test_split
    train_data_t, train_data_e = train_test_split(train_data, test_size=0.2)

    submit = train_data_e[['sid']].copy()

    train_data = train_data_t.drop(['sid', 'pid'], axis=1)
    test_data = train_data_e.drop(['sid', 'pid'], axis=1)

    test_data = test_data.drop(['click_mode'], axis=1)
    train_x = train_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values

    return train_x, train_y, test_data, submit


def gen_plan_new():
    data = merge_raw_data()
    data = gen_od_feas(data)
    plans_df = get_plan_df(data)
    plans_df.to_csv('../data/plans_new.csv')
    return plans_df


def get_train_test_feas_data_2():
    data_all = pd.read_csv('../data/features_new.csv')
    # 排除训练、测试数中没有plan的数据 8946+1787=10733
    data_exclude = data_all[(data_all['click_mode'].isin([0, -1]) & (data_all['max_dist'] == -1))]
    data = data_all[~data_all.sid.isin(data_exclude.sid)]
    submit1 = data.loc[data['click_mode'] == -1, ['sid']].copy()
    submit2 = data_exclude.loc[data_exclude['click_mode'] == -1, ['sid']].copy()
    train_x, train_y, test_x, submit, train_sid, test_sid = split_train_test(data)

    return train_x, train_y, test_x, submit1, submit2, train_sid, test_sid


if __name__ == '__main__':
    import os

    os.chdir('D:/github/recommend/recommend/Context-Aware-Multi-Modal-Transportation-Recommendation-master/code')
    # gen_plan_new()
    gen_train_test_feas_data()
    #
