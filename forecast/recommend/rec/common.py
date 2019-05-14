# -*- coding: utf-8 -*-
num_of_class = 12
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

quantiles_range = np.arange(0, 1.5, 0.5)
stats_feat_num = 5
from geopy.distance import geodesic

def pairwise_cosine_euc_dist(X_test, X_train, dist_metric):
    """
    consine 距离
    euclidean distance 距离
    :param X_test:
    :param X_train:
    :param dist_metric:
    :return:
    """
    if dist_metric == "cosine":
        # 1-余弦距离=余弦相似度
        sim = 1. - pairwise_distances(X_test, X_train, metric=dist_metric, n_jobs=1)
    elif dist_metric == "euclidean":
        # 返回xtest行 xtrain列的array 欧式距离，越大越远；update sqw
        # 欧式距离超过1 不能用1-作为相似度
        sim = pairwise_distances(X_test, X_train, metric=dist_metric, n_jobs=1)
    return sim


def get_sample_indices_by_relevance(dfTrain, additional_key=None):
    """
    labal做key,顺序号做值(0开始)
    {('sid','click_mode'):[]}
    :param dfTrain:
    :param additional_key: qid
    :return:
    """
    # 从零开始编号
    dfTrain["sample_index"] = range(dfTrain.shape[0])
    group_key = ["click_mode"]
    # 按照相关性、qid分组
    if additional_key != None:
        group_key.insert(0, additional_key)
    # 根据相关性分组 每组序号放到[]里;as_index=False agg.index=[0,1,...]
    # as_index=False group_key多个时候无效
    agg = dfTrain.groupby(group_key, as_index=False).apply(lambda x: list(x["sample_index"]))
    # 生成相关性为键的字典
    d = dict(agg)
    dfTrain.drop("sample_index", inplace=True, axis=1)
    return d


def generate_dist_stats_feat(dist_metric, X_train, ids_train, X_test, ids_test, indices_dict, qids_test=None):
    """
    生成每一个test 与train中的各个【类别行】之间的 【距离】的（统计特征）(平均值、方差、最小、中位数、最大)
    如果qids_test不为空，则每个test和与他相同的qid的类别做比较
    :param dist_metric: 距离度量标准 cosine/euclidean
    :param X_train:
    :param ids_train: train 的 train id:注意并不连续
    :param X_test:
    :param ids_test: test id:注意并不连续
    :param indices_dict: 类别键值字典
    :param qids_test: 类别+qid键值字典 ：qid 从1开始
    :return: len(ids_test) 行 stats_feat_num*n_classes列的矩阵---20个列
    每行test-跟-某个类别的train 多对多距离，求距离的 五个统计指标
    每行test-跟-某个类别的train 多对多距离，求距离的 五个统计指标
    stats_func ：全局函数
    stats_feat_num：全局
    """
    # 生成 len(ids_test)行，class(分类个数)列的 多维数组
    stats_feat = np.zeros((len(ids_test), stats_feat_num * num_of_class), dtype=float)
    # 生成距离矩阵
    dis = pairwise_cosine_euc_dist(X_test, X_train, dist_metric)
    for i in range(len(ids_test)):
        # test id :不连续
        id = ids_test[i]
        if qids_test is not None:
            # test qid
            qid = qids_test[i]
        # 一行分别于某一类的距离做比较
        for j in range(num_of_class):
            # (qid,j+1) bug ? sqw 不是as_index=False 吗？
            key = (qid, j + 1) if qids_test is not None else j + 1
            if indices_dict.has_key(key):
                # 找到分组对应的 [0,....] :从0开始的行号
                inds = indices_dict[key]
                # 排除自身 [test_id != train_id]
                inds = [ind for ind in inds if id != ids_train[ind]]
                # 1个test 和 train中某一类别的距离数组[]
                sim_tmp = dis[i][inds]
                if len(sim_tmp) != 0:
                    # 距离的平均值、方差
                    feat = [func(sim_tmp) for func in [np.mean, np.std]]
                    # quantile
                    sim_tmp = pd.Series(sim_tmp)
                    # 距离的最小值、中位数、最大值
                    quantiles = sim_tmp.quantile(quantiles_range)
                    feat = np.hstack((feat, quantiles))
                    # 每一行生成 [五个值的数组]
                    stats_feat[i, j * stats_feat_num:(j + 1) * stats_feat_num] = feat
    return stats_feat

