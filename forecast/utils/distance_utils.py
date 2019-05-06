# coding:utf-8
__author__ = 'songquanwang'
"""
    distance common function
    python种距离计算包括两种方式。
    scipy.spatial.distance里面包括的距离很多
    [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, 
    ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, 
    ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] 
    但是，该方法不支持稀疏矩阵；
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import forecast.utils.utils as utils


def cosine_sim(x, y):
    """
    计算余弦相似性距离(夹角越小，相似度越高)
        cosine_similarity(np.array([1,2,3]).reshape(1,-1),np.array([1,2,3]).reshape(1,-1)) # array([[1.]])
        a=[[1,3],[2,2]]
        cosine_similarity(a)==1-pairwise_distances(a,metric="cosine")
        返回 shape=[len(A),len(B)]矩阵
    :param x:
    :param y:
    :return:
    """
    try:
        # reshape(-1,1) 否则警告 bug
        d = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
        d = d[0][0]
    except:
        print('cosine_sim x is {0} y is {1}'.format(x, y))
        d = 0.
    return d


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


def jaccard_coef(A, B):
    """
    相似度
    杰卡德相似度，用于衡量有限样本集之间的相似度
    :param A:
    :param B:
    :return:
    """
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = utils.try_divide(intersect, union)
    return coef


def dice_coef(A, B):
    """
    Dice 相似度
    2 * intersect/union
    :param A:
    :param B:
    :return:
    """
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = utils.try_divide(2 * intersect, union)
    return d


def compute_jaccard_dice_dist(A, B, dist="jaccard_coef"):
    """
    jaccard 、dice距离
    :param A:
    :param B:
    :param dist:
    :return:
    """
    if dist == "jaccard_coef":
        d = jaccard_coef(A, B)
    elif dist == "dice_coef":
        d = dice_coef(A, B)
    return d


# pairwise distance

def pairwise_jaccard_coef(A, B):
    """
    两个集合之间的Jaccard相似度
    :param A:
    :param B:
    :return:
    """
    coef = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            coef[i, j] = jaccard_coef(A[i], B[j])
    return coef


def pairwise_dice_dist(A, B):
    """
    两个集合之间的Dice相似度
    :param A:
    :param B:
    :return:
    """
    d = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d[i, j] = dice_coef(A[i], B[j])
    return d


def pairwise_jaccard_dice_dist(A, B, dist="jaccard_coef"):
    """
    两个集合之间 dice jaccard相似度的封装函数
    :param A:
    :param B:
    :param dist:
    :return:
    """
    if dist == "jaccard_coef":
        d = pairwise_jaccard_coef(A, B)
    elif dist == "dice_dist":
        d = pairwise_dice_dist(A, B)
    return d
