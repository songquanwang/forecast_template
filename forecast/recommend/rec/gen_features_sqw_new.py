# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-04-17 19:32:26 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-04-17 19:32:26 
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from six.moves import reduce
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


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
    3.te_queries中没有te_plans的占 1787
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
    train:491054 8946
    test: 92571  1787
    all: 583625
    pid:49119    train: 45247 test:12890  3872 个没有在训练数据中
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
    tr_data = tr_data.merge(tr_plans, on='sid', how='inner')
    # click_time无用
    tr_data = tr_data.drop(['click_time'], axis=1)
    # 训练数据：click_mode 置0
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)
    # 测试数据
    te_data = te_queries.merge(te_plans, on='sid', how='inner')
    # 测试数据：click_mode 置-1
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    # data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    data['pid'] = data['pid'].fillna(-1)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def gen_plan_df(data):
    """
    [1, 2, 7, 9, 11] price 存在''
    [3,5,6] price全是0
    4 8 10 price不为''
    对数据中的plans展开，生成plans dataframe
    对plans 进行预处理，填充
    :return:
    """
    # data.loc[data['click_mode'] != -1, 'is_train'] = 1
    # data.loc[data['click_mode'] == -1, 'is_train'] = 0
    data.loc[data['plans'].isnull(), 'plans'] = '[]'
    data['plans'] = data['plans'].apply(lambda line: eval(line))
    lens = [len(item) for item in data['plans']]
    # plan_time distance eta price transport_mode
    sid_list = np.repeat(data['sid'].values, lens)
    plan_time_list = np.repeat(data['plan_time'].values, lens)
    plans = np.concatenate(data['plans'].values)
    plan_pos = np.concatenate([list(range(1, l + 1)) for l in lens])
    df_data = []
    for s, t, p in zip(sid_list, plan_time_list, plans):
        p['sid'] = s
        p['plan_time'] = t
        df_data.append(p)
    # 生成新的plans_df
    plans_df = pd.DataFrame(df_data)
    plans_df = plans_df[['sid', 'plan_time', 'distance', 'eta', 'price', 'transport_mode']]
    plans_df['plan_time'] = pd.to_datetime(plans_df['plan_time'])
    # '' 替换成np.nan
    plans_df['price'] = plans_df['price'].replace(r'', np.NaN)
    plans_df['plan_pos'] = plan_pos

    ###############
    def convert_time(d, m):
        return (d.hour * 60 + d.minute) // m

    # 3 5 6 价格填充为0
    plans_df.loc[plans_df['transport_mode'].isin([3, 5, 6]), 'price'] = 0
    plans_df['time_num30'] = plans_df['plan_time'].apply(lambda x: convert_time(x, 30))
    # 计算单价和mode平均单价
    plans_df['dj'] = plans_df['price'] / plans_df['distance']
    # 单价>3大多是因为距离太近导致，只有60个；只有13条异常数据
    # mode 3 有两条异常数据，估计使收取过路费
    plans_df.loc[(plans_df['transport_mode'] == 4) & (plans_df['dj'] > 3), 'dj'] = 3
    plans_df['mdj'] = plans_df.groupby(['transport_mode', 'time_num30'])['dj'].transform(lambda x: np.nanmedian(x))
    # 填充 price dj[1, 2, 7, 9, 11]
    # 用平均单价替换价格
    plans_df.loc[plans_df['price'].isnull(), 'dj'] = plans_df.loc[plans_df['price'].isnull(), 'mdj']
    df2 = plans_df.loc[plans_df['price'].isnull()]
    # 价格为''的 用单价*距离代替价格
    plans_df.loc[plans_df['price'].isnull(), 'price'] = df2['dj'] * df2['distance']
    # 生成 速度和 速度/单价比
    plans_df['sd'] = plans_df['distance'] / plans_df['eta']
    plans_df['sd_dj'] = plans_df['sd'] / plans_df['dj']
    # sid, plan_time, distance, eta, price, transport_mode ；最高性价比
    plans_df.loc[plans_df['sd_dj'] > 1000, 'sd_dj'] = 1000
    ###
    plans_df['price_eta'] = plans_df['price'] * plans_df['eta']

    return plans_df[['sid', 'plan_time', 'plan_pos', 'distance', 'eta', 'price', 'transport_mode', 'dj', 'sd', 'sd_dj', 'price_eta']]


def gen_cluster(cluster_num, X):
    """
    对地址聚簇
    :param cluster_num:
    :param X:
    :return:
    """
    # 将数据聚类成2个
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    return y_pred, kmeans


def get_dis(x):
    """
    获取经纬度直线距离
    :param o:
    :param d:
    :return:
    """
    o1, o2 = x['o2'], x['o1']
    d1, d2 = x['d2'], x['d1']
    return geodesic((o1, o2), (d1, d2)).m


def add_od_feas(data, cluster_list=[10, 20, 30]):
    """
    经度、维度分开
    :param data:
    :return:
    """
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    # data = data.drop(['o', 'd'], axis=1)
    for num in cluster_list:
        data['o{}'.format(num)], _ = gen_cluster(num, data[['o1', 'o2']].values)
        data['d{}'.format(num)], _ = gen_cluster(num, data[['d1', 'd2']].values)
        data.loc[data['o{}'.format(num)] != data['d{}'.format(num)], 'same_cls{}'.format(num)] = 0
        data.loc[data['o{}'.format(num)] == data['d{}'.format(num)], 'same_cls{}'.format(num)] = 1
        data['o{}_max_mode'.format(num)] = data.groupby(['o{}'.format(num)])['click_mode'].transform(lambda x: mode_max(x.value_counts()))
        data['d{}_max_mode'.format(num)] = data.groupby(['d{}'.format(num)])['click_mode'].transform(lambda x: mode_max(x.value_counts()))
    # 添加距离特征
    data['num_direct_distance'] = data[['o1', 'o2', 'd1', 'd2']].apply(get_dis, axis=1)

    # data['is_rain_max_mode'] = data.groupby(['pid', 'is_rain'])['click_mode'].transform(lambda x: mode_max(x.value_counts()))
    return data


def gen_plan_feas_by_plans(plans_df):
    """
    plans_features = [
                  'mode_feas_0', 'mode_feas_1', 'mode_feas_2', 'mode_feas_3',
                  'mode_feas_4', 'mode_feas_5', 'mode_feas_6', 'mode_feas_7',
                  'mode_feas_8', 'mode_feas_9', 'mode_feas_10', 'mode_feas_11',
                  'first_mode','mode_texts',
                  'max_dist', 'min_dist', 'mean_dist', 'std_dist',
                  'max_price', 'min_price', 'mean_price', 'std_price',
                  'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                  'max_dj', 'min_dj', 'mean_dj', 'std_dj',
                  'max_sd', 'min_sd', 'mean_sd', 'std_sd',
                  'max_sd_dj', 'min_sd_dj', 'mean_sd_dj', 'std_sd_dj',
                  'max_dist_mode', 'min_dist_mode', 'max_price_mode',
                  'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                  'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                  'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                  'svd_mode_9'
                  ]
    max_dist_mode
    :param data:
    :return:
    """
    mode_columns_names = ['mode_feas_{}'.format(i) for i in range(12)]

    def gen_mode_code(mode_list):
        ma = np.zeros(12)
        ma[mode_list] = 1
        return ma

    # 生成 一组计划的mode 占位符
    mode_g = plans_df.groupby('sid')['transport_mode'].apply(gen_mode_code).reset_index()
    mode_columns = ['sid'] + mode_columns_names
    mode_data = np.concatenate(mode_g['transport_mode'].values, axis=0).reshape(len(mode_g), 12)
    sid_data = mode_g['sid'].values.reshape(len(mode_g), 1)
    mode_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)

    def get_first(x):
        return x.values[0]

    def get_second(x):
        if len(x) < 2:
            return x.values[0]
        return x.values[1]

    def get_last(x):
        return x.values[-1]

    def gen_mode_texts(x):
        tl = ' '.join(['word_{}'.format(mode) for mode in x.values])
        return tl

    agg_fun = {'transport_mode': [get_first, get_second, get_last, gen_mode_texts],
               'distance': ['max', 'min', 'mean', lambda x: np.std(x)],
               'price': ['max', 'min', 'mean', lambda x: np.std(x)],
               'eta': ['max', 'min', 'mean', lambda x: np.std(x)],
               # 添加三组特征
               'dj': ['max', 'min', 'mean', lambda x: np.std(x)],
               'sd': ['max', 'min', 'mean', lambda x: np.std(x)],
               'sd_dj': ['max', 'min', 'mean', lambda x: np.std(x)],
               'price_eta': ['max', 'min', 'mean', lambda x: np.std(x)]
               }
    # std ddof =1
    agg_columns = ['sid', 'first_mode', 'second_mode', 'last_mode_1', 'mode_texts',
                   'max_dist', 'min_dist', 'mean_dist', 'std_dist',
                   'max_price', 'min_price', 'mean_price', 'std_price',
                   'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                   'max_dj', 'min_dj', 'mean_dj', 'std_dj',
                   'max_sd', 'min_sd', 'mean_sd', 'std_sd',
                   'max_sd_dj', 'min_sd_dj', 'mean_sd_dj', 'std_sd_dj',
                   'max_price_eta', 'min_price_eta', 'mean_price_eta', 'std_price_eta'
                   ]

    agg_df = plans_df.groupby('sid').agg(agg_fun).reset_index()
    agg_df.columns = agg_columns
    merge_df = pd.merge(plans_df, agg_df, on=['sid'], how='inner')
    # 原来版本是 keep='last'
    # dist
    max_dist_mode_df = merge_df.loc[merge_df['distance'] == merge_df['max_dist'], ['sid', 'transport_mode']]
    max_dist_mode_df.columns = ['sid', 'max_dist_mode']
    max_dist_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    min_dist_mode_df = merge_df.loc[merge_df['distance'] == merge_df['min_dist'], ['sid', 'transport_mode']]
    min_dist_mode_df.columns = ['sid', 'min_dist_mode']
    min_dist_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    # price
    max_price_mode_df = merge_df.loc[merge_df['price'] == merge_df['max_price'], ['sid', 'transport_mode']]
    max_price_mode_df.columns = ['sid', 'max_price_mode']
    max_price_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    min_price_mode_df = merge_df.loc[merge_df['price'] == merge_df['min_price'], ['sid', 'transport_mode']]
    min_price_mode_df.columns = ['sid', 'min_price_mode']
    min_price_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    # eta
    max_eta_mode_df = merge_df.loc[merge_df['eta'] == merge_df['max_eta'], ['sid', 'transport_mode']]
    max_eta_mode_df.columns = ['sid', 'max_eta_mode']
    max_eta_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    min_eta_mode_df = merge_df.loc[merge_df['eta'] == merge_df['min_eta'], ['sid', 'transport_mode']]
    min_eta_mode_df.columns = ['sid', 'min_eta_mode']
    min_eta_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)

    # sd
    max_sd_mode_df = merge_df.loc[merge_df['sd'] == merge_df['max_sd'], ['sid', 'transport_mode']]
    max_sd_mode_df.columns = ['sid', 'max_sd_mode']
    max_sd_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)

    min_sd_mode_df = merge_df.loc[merge_df['sd'] == merge_df['min_sd'], ['sid', 'transport_mode']]
    min_sd_mode_df.columns = ['sid', 'min_sd_mode']
    min_sd_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    # dj
    max_dj_mode_df = merge_df.loc[merge_df['dj'] == merge_df['max_dj'], ['sid', 'transport_mode']]
    max_dj_mode_df.columns = ['sid', 'max_dj_mode']
    max_dj_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)

    min_dj_mode_df = merge_df.loc[merge_df['dj'] == merge_df['min_dj'], ['sid', 'transport_mode']]
    min_dj_mode_df.columns = ['sid', 'min_dj_mode']
    min_dj_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
    # sj_dj
    max_sd_dj_mode_df = merge_df.loc[merge_df['sd_dj'] == merge_df['max_sd_dj'], ['sid', 'transport_mode']]
    max_sd_dj_mode_df.columns = ['sid', 'max_sd_dj_mode']
    max_sd_dj_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)

    min_sd_dj_mode_df = merge_df.loc[merge_df['sd_dj'] == merge_df['min_sd_dj'], ['sid', 'transport_mode']]
    min_sd_dj_mode_df.columns = ['sid', 'min_sd_dj_mode']
    min_sd_dj_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
    # price eta
    max_price_eta_mode_df = merge_df.loc[merge_df['price_eta'] == merge_df['max_price_eta'], ['sid', 'transport_mode']]
    max_price_eta_mode_df.columns = ['sid', 'max_price_eta_mode']
    max_price_eta_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)

    min_price_eta_mode_df = merge_df.loc[merge_df['price_eta'] == merge_df['min_price_eta'], ['sid', 'transport_mode']]
    min_price_eta_mode_df.columns = ['sid', 'min_price_eta_mode']
    min_price_eta_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)

    complex_feature_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['sid'], how='inner'),
                                [max_dist_mode_df, min_dist_mode_df,
                                 max_price_mode_df, min_price_mode_df,
                                 max_eta_mode_df, min_eta_mode_df,
                                 max_sd_mode_df, min_sd_mode_df,
                                 max_dj_mode_df, min_dj_mode_df,
                                 max_sd_dj_mode_df, min_sd_dj_mode_df,
                                 max_price_eta_mode_df, min_price_eta_mode_df])
    plan_feature_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['sid'], how='inner'),
                             [mode_df, agg_df, complex_feature_df])

    return plan_feature_df


def gen_empty_plan_feas(data):
    """
    生成empty plans
      plans_features = [
                  'mode_feas_0', 'mode_feas_1', 'mode_feas_2', 'mode_feas_3',
                  'mode_feas_4', 'mode_feas_5', 'mode_feas_6', 'mode_feas_7',
                  'mode_feas_8', 'mode_feas_9', 'mode_feas_10', 'mode_feas_11',
                  'first_mode','mode_texts',
                  'max_dist', 'min_dist', 'mean_dist', 'std_dist',
                  'max_price', 'min_price', 'mean_price', 'std_price',
                  'max_eta', 'min_eta', 'mean_eta', 'std_eta',
                  'max_dj', 'min_dj', 'mean_dj', 'std_dj',
                  'max_sd', 'min_sd', 'mean_sd', 'std_sd',
                  'max_sd_dj', 'min_sd_dj', 'mean_sd_dj', 'std_sd_dj',
                  'max_dist_mode', 'min_dist_mode', 'max_price_mode',
                  'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                  'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3',
                  'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8',
                  'svd_mode_9'

                  ]
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
    plan_feature_df['second_mode'] = 0
    plan_feature_df['last_mode_1'] = 0
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

    # 新增特征
    plan_feature_df['max_dj'] = -1
    plan_feature_df['min_dj'] = -1
    plan_feature_df['mean_dj'] = -1
    plan_feature_df['std_dj'] = -1

    plan_feature_df['max_sd'] = -1
    plan_feature_df['min_sd'] = -1
    plan_feature_df['mean_sd'] = -1
    plan_feature_df['std_sd'] = -1

    plan_feature_df['max_sd_dj'] = -1
    plan_feature_df['min_sd_dj'] = -1
    plan_feature_df['mean_sd_dj'] = -1
    plan_feature_df['std_sd_dj'] = -1

    plan_feature_df['max_price_eta'] = -1
    plan_feature_df['min_price_eta'] = -1
    plan_feature_df['mean_price_eta'] = -1
    plan_feature_df['std_price_eta'] = -1

    plan_feature_df['max_dist_mode'] = -1
    plan_feature_df['min_dist_mode'] = -1
    plan_feature_df['max_price_mode'] = -1
    plan_feature_df['min_price_mode'] = -1
    plan_feature_df['max_eta_mode'] = -1
    plan_feature_df['min_eta_mode'] = -1
    # 新增
    plan_feature_df['max_sd_mode'] = -1
    plan_feature_df['min_sd_mode'] = -1
    plan_feature_df['max_dj_mode'] = -1
    plan_feature_df['min_dj_mode'] = -1
    plan_feature_df['max_sd_dj_mode'] = -1
    plan_feature_df['min_sd_dj_mode'] = -1
    plan_feature_df['max_price_eta_mode'] = -1
    plan_feature_df['min_price_eta_mode'] = -1

    return plan_feature_df


def gen_plan_feas(data):
    """
    计划特征 [max min mean std] * 3 + 8 mode
    :param data:
    :return:
    """
    # plans_df = get_plan_df(data)
    # tr_plans + te_plans =583625
    plans_df = pd.read_csv('../data/data_set_phase1/plans_djsd.csv')
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


def get_plan_feas():
    """
    获取
    :return:
    """
    plan_featurns = pd.read_csv('../data/data_set_phase1/plan_features.csv')
    return plan_featurns


def add_sta_feas(data):
    """
    添加统计特征
    计算 由svd_mode 代表的 推荐计划列表 分别于各个label 向量中心的距离
    最大距离、最小距离、平均距离、方差

    svd mode 10维
    :param data:
    :return:
    """
    svd_cloumns = ['svd_mode_{}'.format(i) for i in range(10)]
    mode_center_dis_columns = ['mode_center_dis_{}'.format(i) for i in range(12)]
    agg_df = data.groupby('click_mode')[svd_cloumns].agg(np.mean).reset_index().sort_values(by='click_mode').values
    dis_matrix = pairwise_distances(data[svd_cloumns], agg_df[svd_cloumns], metric='cosine', n_jobs=1)

    data[mode_center_dis_columns] = dis_matrix.values
    #
    stat_columns = ['max_mode_center_dis', 'min_mode_center_dis', 'mean_center_dis', 'std_center_dis']
    data[stat_columns] = np.max(dis_matrix.values, axis=1), np.min(dis_matrix.values, axis=1), np.mean(dis_matrix.values, axis=1), np.std(dis_matrix.values, axis=1)
    return data


def add_profile_feas(data):
    """
    用户特征 20维度
    :param data:
    :return:
    """
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data = data.merge(svd_feas, on='pid', how='left')
    return data


def add_time_feas(data):
    """
    时间特征
    :param data:
    :return:
    """
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    data = data.drop(['req_time'], axis=1)
    return data


def gen_plan_new():
    """
    预处理plans,保存成文件
    :return:
    """
    data = merge_raw_data()
    data = add_od_feas(data)
    plans_df = gen_plan_df(data)
    plans_df.to_csv('../data/data_set_phase1/plans_djsd.csv')
    return plans_df


# 需要排除-1,否则无法对测试集构造此特征

def mode_max(c):
    return c[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]].idxmax()
    # return c.idxmax()


def add_is_rain(data):
    """
    生成天气特征
    :return:
    """
    weather_df = pd.read_csv('../data/data_set_phase1/weather_date.csv', parse_dates=['dt'])
    data['plan_time'] = pd.to_datetime(data['plan_time'])
    data.loc[data['plan_time'].notnull(), 'dt_str'] = data.loc[data['plan_time'].notnull(), 'plan_time'].apply(lambda x: x.strftime('%Y-%m-%d %H'))
    weather_df['dt_str'] = weather_df['dt'].apply(lambda x: x.strftime('%Y-%m-%d %H'))
    weather_df['is_rain'] = 0
    weather_df.loc[weather_df['weather'] == '阵雨', 'is_rain'] = 1
    weather_df.to_csv('../data/data_set_phase1/weather_date_expanded.csv')
    merge_df = pd.merge(data, weather_df[['dt_str', 'is_rain']], on=['dt_str'], how='left')
    merge_df = merge_df.drop(['dt_str'], axis=1)
    merge_df.loc[merge_df['is_rain'].isnull(), 'is_rain'] = 0

    # 会有Nan值 需要排除-1,否则无法对测试集构造此特征
    merge_df['is_rain_max_mode'] = merge_df.groupby(['pid', 'is_rain'])['click_mode'].transform(lambda x: mode_max(x.value_counts()))
    return merge_df


def gen_plan_extra_features(data, plan_df):
    """
    'pid', 'pid_max_mode',
    'pid_max_dist', 'pid_min_dist', 'pid_mean_dist',
    'pid_std_dist', 'pid_max_price', 'pid_min_price', 'pid_mean_price',
    'pid_std_price', 'pid_max_eta', 'pid_min_eta', 'pid_mean_eta',
    'pid_std_eta', 'pid_max_dj', 'pid_min_dj', 'pid_mean_dj', 'pid_std_dj',
    'pid_max_sd', 'pid_min_sd', 'pid_mean_sd', 'pid_std_sd',
    'pid_max_sd_dj', 'pid_min_sd_dj', 'pid_mean_sd_dj', 'pid_std_sd_dj',
    'mode_num_0', 'mode_num_1', 'mode_num_2', 'mode_num_3', 'mode_num_4',
    'mode_num_5', 'mode_num_6', 'mode_num_7', 'mode_num_8', 'mode_num_9',
    'mode_num_10', 'mode_num_11

    len(data.loc[~data['click_mode'].isin([0,-1]),'sid']) # 453336
    :param data:
    :return:返回 pid维度数据
    """
    plan_df = pd.read_csv('../data/data_set_phase1/plans_djsd.csv')
    # 用户点击量统计；去掉测试数据中-1 583625  ;train:45247
    cut_df = data[['sid', 'pid', 'first_mode']]
    # 2706625 - 110208 重复记录 2596417
    cut_plan_df = plan_df[['sid', 'plan_pos', 'distance', 'eta', 'price', 'transport_mode', 'dj', 'sd', 'sd_dj']]
    # 去挑一个sid 对应两个相同的mode 的plan 2596417
    cut_plan_df = cut_plan_df.drop_duplicates(subset=['sid', 'transport_mode'])
    # 虽然推荐了，用户没有点击的（click_mode =0） 的记录没有关联上   sid:37718  pid: 11772
    merge_df = pd.merge(cut_df, cut_plan_df, left_on=['sid', 'first_mode'], right_on=['sid', 'transport_mode'], how='left')
    # 填充-1就错了
    # merge_df.loc[merge_df['plan_pos'].isnull(), ['plan_pos', 'distance', 'eta', 'price', 'dj', 'sd', 'sd_dj']] = -1
    merge_df.loc[merge_df['transport_mode'].isnull(), 'transport_mode'] = 0
    # 生成聚合属性
    mode_num_names = ['mode_num_{}'.format(i) for i in range(12)]
    # pandas 自动忽略np.nan
    agg_fun = {
        'first_mode': [lambda x: x.value_counts().idxmax()],
        'distance': ['max', 'min', 'mean', lambda x: np.std(x)],
        'price': ['max', 'min', 'mean', lambda x: np.std(x)],
        'eta': ['max', 'min', 'mean', lambda x: np.std(x)],
        'dj': ['max', 'min', 'mean', lambda x: np.std(x)],
        'sd': ['max', 'min', 'mean', lambda x: np.std(x)],
        'sd_dj': ['max', 'min', 'mean', lambda x: np.std(x)]
    }
    # 统计数字会出现 -1 ，一个用户 有时候点击，有时候不点击，就会出现-1
    agg_columns = ['pid',
                   'pid_max_mode',
                   'pid_max_dist', 'pid_min_dist', 'pid_mean_dist', 'pid_std_dist',
                   'pid_max_price', 'pid_min_price', 'pid_mean_price', 'pid_std_price',
                   'pid_max_eta', 'pid_min_eta', 'pid_mean_eta', 'pid_std_eta',
                   'pid_max_dj', 'pid_min_dj', 'pid_mean_dj', 'pid_std_dj',
                   'pid_max_sd', 'pid_min_sd', 'pid_mean_sd', 'pid_std_sd',
                   'pid_max_sd_dj', 'pid_min_sd_dj', 'pid_mean_sd_dj', 'pid_std_sd_dj'
                   ]
    pid_feature_df = merge_df.groupby('pid').agg(agg_fun).reset_index()
    pid_feature_df.columns = agg_columns

    def mode_num(c):
        """

        :param c:
        :return:
        """
        z = np.zeros(12)
        k = c.index.values.astype(np.int32)
        v = c.values
        idx = np.where(k > -1)
        kc = k[idx]
        vc = v[idx]
        z[kc] = vc
        return z / np.sum(z)

    pid_group_df = merge_df.groupby('pid')['first_mode'].apply(lambda x: mode_num(x.value_counts())).reset_index()

    mode_columns = ['pid'] + mode_num_names

    mode_data = np.concatenate(pid_group_df['first_mode'].values, axis=0).reshape(len(pid_group_df), 12)
    sid_data = pid_group_df['pid'].values.reshape(len(pid_group_df), 1)
    mode_num_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)
    mode_num_df.columns = mode_columns
    pid_ext_features_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['pid'], how='inner'), [pid_feature_df, mode_num_df])
    # 2904 存在空值 把nan 填充 -1
    pid_ext_features_df.loc[pid_ext_features_df['pid_max_dist'].isnull(), agg_columns[2:]] = -1
    # 占比也改成-1  没有意义
    pid_ext_features_df.loc[pid_ext_features_df['pid_max_dist'] == -1, mode_num_names] = -1
    # 为没有关联上的pid 构造默认值 3872 个pid  没有特征 np.nan
    # pid_list = list(set(data.pid) - set(pid_ext_features_df['pid']))
    #
    # # default_values = pid_ext_features_df[pid_ext_features_df['pid'] == -1].iloc[[0]]
    # mean_series = pid_ext_features_df.mean()
    # default_values = pd.DataFrame(mean_series.values.reshape(1, -1), columns=mean_series.index)
    # default_values['pid_max_mode'] = pid_ext_features_df['pid_max_mode'].median()
    # default_df = pd.concat([default_values] * len(pid_list), axis=0)
    # default_df['pid'] = pid_list
    # pid_ext_features_df = pd.concat([pid_ext_features_df, default_df], axis=0)
    # 保存文件
    round(pid_ext_features_df, 7).to_csv('../data/data_set_phase1/pid_ext_features.csv', index=False)
    return pid_ext_features_df


def get_plan_extra_features():
    """
    获取用户维度的特征数据
    :return:
    """
    plan_ext_features = pd.read_csv('../data/data_set_phase1/pid_ext_features.csv')
    return plan_ext_features


def gen_train_test_feas_data():
    """
    :return:
    """
    # 添加统计特征
    # stat_columns_sid = ['stat_{0}'.format(c) for c in ['min', 'median', 'max', 'mean', 'std']]
    # pid_stat_columns_sid = ['stat_pid_{0}'.format(c) for c in ['min', 'median', 'max', 'mean', 'std']]

    data = merge_raw_data()
    # 添加天气特征
    # data = add_is_rain(data)
    print('add od')
    data = add_od_feas(data)
    print('add plan feat')
    plans_features = gen_plan_feas(data)
    # union没有plans的 innner=left
    data = pd.merge(data, plans_features, on=['sid'], how='left')
    print('add profile feat')
    data = add_profile_feas(data)
    print('add time feat')
    data = add_time_feas(data)
    print('add extra plan feat')
    pid_ext_features_df = gen_plan_extra_features(data, None)
    data = pd.merge(data, pid_ext_features_df, on=['pid'], how='left')
    #
    data = data.drop(['plans'], axis=1)
    data = data.drop(['is_train'], axis=1)
    data = data.drop(['mode_texts'], axis=1)
    round(data, 7).to_csv('../data/features/features_all_new.csv', index=False)
    return data


def process_label_imbalance(raw_df):
    """
    处理label不均衡问题
    用了不好
    :param raw_df:
    :return:
    """
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 3]] * 3, ignore_index=True)
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 4]] * 4, ignore_index=True)
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 8]] * 10, ignore_index=True)
    raw_df = raw_df.append([raw_df[raw_df['click_mode'] == 11]] * 4, ignore_index=True)
    return raw_df


def process_ts_feat():
    """
    时序特征
    Index(['sid', 'pid', 'plan_time', 'click_mode', 'last_mode', 'pre_mode',
       'pid_max_dist', 'pid_min_dist', 'pid_mean_dist', 'pid_std_dist',
       'pid_max_price', 'pid_min_price', 'pid_mean_price', 'pid_std_price',
       'pid_max_eta', 'pid_min_eta', 'pid_mean_eta', 'pid_std_eta',
       'pid_max_dj', 'pid_min_dj', 'pid_mean_dj', 'pid_std_dj', 'pid_max_sd',
       'pid_min_sd', 'pid_mean_sd', 'pid_std_sd', 'pid_max_sd_dj',
       'pid_min_sd_dj', 'pid_mean_sd_dj', 'pid_std_sd_dj', 'shift_cm',
       'pid_max_mode', 'mode_num_0', 'mode_num_1', 'mode_num_2', 'mode_num_3',
       'mode_num_4', 'mode_num_5', 'mode_num_6', 'mode_num_7', 'mode_num_8',
       'mode_num_9', 'mode_num_10', 'mode_num_11']
    :return:
    """
    data = merge_raw_data()
    plan_df = pd.read_csv('../data/data_set_phase1/plans_djsd.csv')
    data['plan_time'] = pd.to_datetime('plan_time')
    order_data = data.sort_values(by='plan_time')
    order_data['last_mode'] = order_data.groupby('pid')['click_mode'].transform(lambda x: x.shift(1))
    order_data['pre_mode'] = order_data.groupby('pid')['click_mode'].transform(lambda x: x.shift(1).rolling(3).median())
    ###
    cut_plan_df = plan_df[['sid', 'plan_pos', 'distance', 'eta', 'price', 'transport_mode', 'dj', 'sd', 'sd_dj']]
    # 去挑一个sid 对应两个相同的mode 的plan 2596417
    cut_plan_df = cut_plan_df.drop_duplicates(subset=['sid', 'transport_mode'])
    # 虽然推荐了，用户没有点击的（click_mode =0） 的记录没有关联上   sid:37718  pid: 11772
    merge_df = pd.merge(data, cut_plan_df, left_on=['sid', 'click_mode'], right_on=['sid', 'transport_mode'], how='left')
    merge_df = merge_df.sort_values(by='plan_time')

    merge_df['pid_max_dist'] = merge_df.groupby('pid')['distance'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).max())
    merge_df['pid_min_dist'] = merge_df.groupby('pid')['distance'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).min())
    merge_df['pid_mean_dist'] = merge_df.groupby('pid')['distance'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
    merge_df['pid_std_dist'] = merge_df.groupby('pid')['distance'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())

    merge_df['pid_max_price'] = merge_df.groupby('pid')['price'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).max())
    merge_df['pid_min_price'] = merge_df.groupby('pid')['price'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).min())
    merge_df['pid_mean_price'] = merge_df.groupby('pid')['price'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
    merge_df['pid_std_price'] = merge_df.groupby('pid')['price'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())

    merge_df['pid_max_eta'] = merge_df.groupby('pid')['eta'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).max())
    merge_df['pid_min_eta'] = merge_df.groupby('pid')['eta'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).min())
    merge_df['pid_mean_eta'] = merge_df.groupby('pid')['eta'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
    merge_df['pid_std_eta'] = merge_df.groupby('pid')['eta'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())

    merge_df['pid_max_dj'] = merge_df.groupby('pid')['dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).max())
    merge_df['pid_min_dj'] = merge_df.groupby('pid')['dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).min())
    merge_df['pid_mean_dj'] = merge_df.groupby('pid')['dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
    merge_df['pid_std_dj'] = merge_df.groupby('pid')['dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())

    merge_df['pid_max_sd'] = merge_df.groupby('pid')['sd'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).max())
    merge_df['pid_min_sd'] = merge_df.groupby('pid')['sd'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).min())
    merge_df['pid_mean_sd'] = merge_df.groupby('pid')['sd'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
    merge_df['pid_std_sd'] = merge_df.groupby('pid')['sd'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())

    merge_df['pid_max_sd_dj'] = merge_df.groupby('pid')['sd_dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).max())
    merge_df['pid_min_sd_dj'] = merge_df.groupby('pid')['sd_dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).min())
    merge_df['pid_mean_sd_dj'] = merge_df.groupby('pid')['sd_dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
    merge_df['pid_std_sd_dj'] = merge_df.groupby('pid')['sd_dj'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).std())

    ############
    def gen_pre_list(x):
        le = x.size
        pre_list = list(map(list, zip(*[x.shift(i).values for i in range(1, 1 + le)][::-1])))
        ls = {'shift_cm': pd.Series(pre_list)}
        df = pd.DataFrame(ls, columns=['shift_cm'])
        df.index = x.index
        return df

    # 需要去掉shift_cm里面的 -1
    merge_df.loc[merge_df['pid'] == -1, 'shift_cm'] = np.nan
    merge_df.loc[merge_df['pid'] != -1, 'shift_cm'] = merge_df.loc[merge_df['pid'] != -1].groupby('pid')['click_mode'].apply(gen_pre_list)

    def mode_num(x):
        """
        :param c:
        :return:
        """
        if x is np.nan:
            return [0] * 12
        if -1 in x:
            x = x[0:x.index(-1)]
        c = pd.value_counts(x)
        z = np.zeros(12)
        if len(c) > 0:
            k = c.index.values.astype(np.int32)
            v = c.values
            z[k] = v
            z = z / np.sum(z)
        return z

    mode_num_names = ['mode_num_{}'.format(i) for i in range(12)]
    pid_group_df = merge_df['shift_cm'].apply(lambda x: mode_num(x)).reset_index()
    mode_columns = ['sid'] + mode_num_names
    mode_data = np.concatenate(pid_group_df['shift_cm'].values, axis=0).reshape(len(pid_group_df), 12)
    sid_data = merge_df['sid'].values.reshape(len(merge_df), 1)
    mode_num_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)
    mode_num_df.columns = mode_columns
    merge_df = pd.merge(merge_df, mode_num_df, on=['sid'], how='left')

    def get_max_fre(x):
        if x is np.nan:
            return np.nan
        if -1 in x:
            x = x[0:x.index(-1)]
        c = pd.value_counts(x)
        if len(c) == 0:
            return np.nan
        else:
            return c.idxmax()

    merge_df['pid_max_mode'] = merge_df['shift_cm'].apply(get_max_fre)
    merge_df.to_csv('../data/data_set_phase1/pid_ext_ts_feature.csv', index=False)


# merge_df['num_direct_distance'] = merge_df.apply(lambda x: get_dis(x['o'],x['d']), axis=1)
def pre_next_n(x, n):
    """
    next N；pre N
    :param x:
    :param n:
    param d:1 pre -1 next
    :return:
    """
    return map(list, zip(*[x.shift(i).values for i in range(1, 1 + n)][::-1]))


if __name__ == '__main__':
    # import os

    # os.chdir('D:/github/forecast_template/recommend/rec')
    # gen_plan_new()
    gen_train_test_feas_data()
