# coding:utf-8
"""
__file__

    utils.py

__description__

    This file provides functions for
        1. various customized objectives used together with XGBoost
        2. corresponding decoding method for various objectives
            - MSE
            - Pairwise ranking
            - Softmax
            - Softkappa
            - EBC
            - COCR

__author__

    songquanwang

"""

import numpy as np

from forecast.utils.ml_metrics import quadratic_weighted_kappa
import forecast.conf.model_params_conf as config


######################
## Helper Functions ##
######################
#### sigmoid
def sigmoid(score):
    """
    返回sigmoid 值
    :param score: 可以是向量 也可以是一个值；但不可以直接是数组[]
    :return:
    """
    p = 1. / (1. + np.exp(-score))
    return p


####
def softmax(score):
    """
    计算softmax值
    注意：np.max([],axis=0) 默认是整个数组最大值
    :param score:
     score格式
     array(
       [
           [-0.66825223,  0.2887553 ,  3.40399051,  1.01323175],
           [-0.07798982,  0.44937038,  1.42988169,  0.74605137],
           [-0.62874174,  0.29706955,  3.33471489,  1.01480103]

       ])
    :return:
    array([
       [ 0.0147785 ,  0.03848163,  0.86732722,  0.07941264],
       [ 0.10536016,  0.17852789,  0.47592302,  0.24018893],
       [ 0.01630355,  0.04114877,  0.85820185,  0.08434584]
    ])
    """
    score = np.asarray(score, dtype=float)
    # 下面这行代码无用
    # score = np.exp(score - np.max(score))
    # 每行相加 归一化
    score /= np.sum(score, axis=1)[:, np.newaxis]
    return score


##########################
## Cutomized Objectives ##
##########################
####
def ebcObj(preds, dtrain):
    """
    ebc:Extended Binary Classification
    有序回归 实现了一下论文
    论文：Ordinal Regression by Extended Binary Classification
    :param preds:
    :param dtrain:
    :return:
    grad:一阶梯度
    hess:二阶梯度
    """
    ## label are +1/-1
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    ## extended samples within the feature construction part
    if np.min(labels) == -1 and np.max(labels) == 1:
        s = np.exp(labels * preds)
        grad = - weights * labels / (1. + s)
        hess = weights * (labels ** 2) * s / ((1. + s) ** 2)
        ## TODO: figure out how to apply sample weights
    ## extended samples within the objective value computation part
    else:
        ## label are in [0,1,2,3]
        labels += 1
        M = preds.shape[0]
        N = preds.shape[1]
        grad = np.zeros((M, N), dtype=float)
        hess = np.zeros((M, N), dtype=float)
        ## we only use the first K-1 class for extended examples
        for c in range(N - 1):
            k = c + 1
            Y = 2. * np.asarray(labels > k, dtype=float) - 1.
            C_yk = np.power(Y - k, 2)
            C_yk1 = np.power(Y - (k + 1), 2)
            w = np.abs(C_yk - C_yk1)
            p = preds[:, c]
            s = np.exp(Y * p)
            grad[:, c] = - w * Y / (1. + s)
            hess[:, c] = w * (Y ** 2) * s / ((1. + s) ** 2)
        ## apply sample weights
        grad *= weights[:, np.newaxis]
        hess *= weights[:, np.newaxis]
        # shape改成1维
        grad.shape = (M * N)
        hess.shape = (M * N)
    return grad, hess


#### Implement the method described in the paper:
# Improving ranking performance with cost-sensitive ordinal classification via regression
# Yu-Xun Ruan, Hsuan-Tien Lin, and Ming-Feng Tsai
def cocrObj(preds, dtrain):
    """
    cocr:cost-sensitive ordinal classification via regression
    实现论文：Improving ranking performance with cost-sensitive ordinal classification via regression
    Yu-Xun Ruan, Hsuan-Tien Lin, and Ming-Feng Tsai
    :param preds:
    :param dtrain:
    :return:
    """
    ## label are in [0,1,2,3]
    Y = dtrain.get_label()
    Y = Y[:, np.newaxis]
    ## get sample weights
    weights = dtrain.get_weight()
    weights = weights[:, np.newaxis]
    ##
    M, N = preds.shape
    k = np.asarray(range(1, N + 1))
    k = k[np.newaxis, :]
    b = np.asarray(Y >= k)
    C_yk = np.power(Y - k, 2)
    C_yk1 = np.power(Y - (k - 1), 2)
    w = np.abs(C_yk - C_yk1)
    grad = 2 * w * (preds - b)
    hess = 2 * w

    ## apply sample weights
    grad *= weights
    hess *= weights
    grad.shape = (M * N)
    hess.shape = (M * N)
    return grad, hess


#### directly optimized kappa (old version)
def softkappaObj(preds, dtrain):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]

    ## compute O (enumerator)
    O = 0.0
    for j in range(N):
        wj = (labels - (j + 1.)) ** 2
        O += np.sum(wj * preds[:, j])

    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    hist_pred = np.sum(preds, axis=0)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j + 1.)) ** 2) * preds[:, n] * (indicator - preds[:, j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                dE += pow(k - l, 2.0) * hist_label[l] * preds[:, n] * (indicator - preds[:, k])
        ## the grad
        grad[:, n] = -M * (dO * E - O * dE) / (E ** 2)

        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j + 1.)) ** 2) * preds[:, n] * (1 - 2. * preds[:, n]) * (indicator - preds[:, j])

        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                d2E += pow(k - l, 2.0) * hist_label[l] * preds[:, n] * (1 - 2. * preds[:, n]) * (
                    indicator - preds[:, k])
        ## the hess
        hess[:, n] = -M * ((d2O * E - O * d2E) * (E ** 2) - (dO * E - O * dE) * 2. * E * dE) / (E ** 4)

    grad *= -1.
    hess *= -1.
    # use a const
    # hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = 0.000125 / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess)  # It works!! no idea...
    grad.shape = (M * N)
    hess.shape = (M * N)
    return grad, hess


#### directly optimized kappa (final version)
# since we use the cdf for finding cutoff which results in the same distribution between training/validaion
# so the denominator is kind of fixed
def softkappaObj(preds, dtrain, hess_scale=0.000125):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]
    weights = dtrain.get_weight()
    ## compute O (enumerator)
    # 误差大小
    O = 0.0
    for j in range(N):
        wj = (labels - (j + 1.)) ** 2
        O += np.sum(wj * preds[:, j])

    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    # hist_pred = np.sum(preds, axis=0)
    hist_pred = hist_label
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j + 1.)) ** 2) * preds[:, n] * (indicator - preds[:, j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        # for k in range(N):
        #    for l in range(N):
        #        indicator = float(n == k)
        #        dE += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (indicator - preds[:,k])
        ## the grad
        grad[:, n] = -M * (dO * E - O * dE) / (E ** 2)

        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j + 1.)) ** 2) * preds[:, n] * (1 - 2. * preds[:, n]) * (indicator - preds[:, j])

        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        # for k in range(N):
        #    for l in range(N):
        #        indicator = float(n == k)
        #        d2E += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,k])
        ## the hess
        hess[:, n] = -M * ((d2O * E - O * d2E) * (E ** 2) - (dO * E - O * dE) * 2. * E * dE) / (E ** 4)

    grad *= -1.
    hess *= -1.
    # use a const
    # hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = hess_scale / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess)  # It works!! no idea...
    ## apply sample weights
    grad *= weights[:, np.newaxis]
    hess *= weights[:, np.newaxis]
    grad.shape = (M * N)
    hess.shape = (M * N)
    return grad, hess


#####################
## Decoding Method ##
#####################
#### decoding method for ranking and regression
# cdf array([ 0.07348703,  0.22564841,  0.38818444,  1.        ]) 对pred 由小到大排序索引后，按照 cdf比例 对 pred 进行赋值 1 2 3 4
def getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([4] * num, dtype=int)
    # 排序
    rank = pred.argsort()
    output[rank[:int(num * cdf[0] - 1)]] = 1
    output[rank[int(num * cdf[0]):int(num * cdf[1] - 1)]] = 2
    output[rank[int(num * cdf[1]):int(num * cdf[2] - 1)]] = 3
    if valid:
        cutoff = [pred[rank[int(num * cdf[i] - 1)]] for i in range(3)]
        return output, cutoff
    return output


#### get test score using cutoff found in the validation set
def getTestScore(pred, cutoff):
    num = pred.shape[0]
    output = np.asarray([4] * num, dtype=int)
    for i in range(num):
        if pred[i] <= cutoff[0]:
            output[i] = 1
        elif pred[i] <= cutoff[1]:
            output[i] = 2
        elif pred[i] <= cutoff[2]:
            output[i] = 3
    return output


#### decoding method for four class probabilities (e.g., softmax classification) 没用到
def getClfScore(preds, cdf):
    w = np.asarray(np.arange(1, config.num_of_class + 1))
    preds = preds * w[np.newaxis, :]
    preds = np.sum(preds, axis=1)
    output = getScore(preds, cdf)
    output = np.asarray(output, dtype=int)
    return output


#### decoding method for EBC
def applyEBCRule(preds, hard_threshold=False):
    if len(preds.shape) == 1:
        ## get prediction
        numOfSample = len(preds) / (config.num_of_class - 1)
        if hard_threshold:
            r = np.ones((numOfSample), dtype=int)
        else:
            r = np.ones((numOfSample), dtype=float)
        for c in range(config.num_of_class - 1):
            if hard_threshold:
                r += np.asarray(preds[c * numOfSample:(c + 1) * numOfSample] > 0, dtype=int)
            else:
                r += preds[c * numOfSample:(c + 1) * numOfSample]
    elif preds.shape[1] == 4:
        if hard_threshold:
            r = np.sum(np.asarray(preds[:, :3] > 0), axis=1) + 1
        else:
            r = np.sum(preds[:, :3], axis=1) + 1
    return r


#### decoding method for COCR
def applyCOCRRule(preds):
    r = np.sum(preds, axis=1) + 1
    return r


#############################################
## Cutomized Evaluation Metric for XGBoost ##
#############################################
#### evalerror for regression and pairwise ranking
def evalerror_regrank_cdf(preds, dtrain, cdf):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    preds = getScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)


#### evalerror for softmax
def evalerror_softmax_cdf(preds, dtrain, cdf):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    preds = getClfScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)


#### evalerror for soft-kappa
def evalerror_softkappa_cdf(preds, dtrain, cdf):
    ## label are in [0,1,2,3]
    labels = dtrain.get_label() + 1
    preds = softmax(preds)
    preds = getClfScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)


#### evalerror for EBC
def evalerror_ebc_cdf(preds, dtrain, cdf, hard_threshold=False):
    labels = dtrain.get_label()
    ## extended samples within the feature construction part
    if np.min(labels) == -1 and np.max(labels) == 1:
        labels = applyEBCRule(labels)
    ## extended samples within the objective value computation part
    ## See ebcobj function for detail
    else:
        ## label are in [0,1,2,3]
        labels += 1
    # print preds.shape
    ## get prediction
    # hard = False
    if hard_threshold:
        preds = applyEBCRule(preds, hard_threshold=hard_threshold)
    else:
        preds = sigmoid(preds)
        preds = applyEBCRule(preds, hard_threshold=hard_threshold)
        preds = getScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)


#### evalerror for COCR
def evalerror_cocr_cdf(preds, dtrain, cdf):
    labels = dtrain.get_label() + 1
    # print preds.shape
    ## get prediction
    # preds = sigmoid(preds)
    preds = applyCOCRRule(preds)
    preds = getScore(preds, cdf)
    kappa = quadratic_weighted_kappa(labels, preds)
    ## we return -kappa for using early stopping
    kappa *= -1.
    return 'kappa', float(kappa)


def bootstrap_all(bootstrap_replacement, numTrain, bootstrap_ratio):
    """
    对全部训练数据进行自举法抽样
    :param bootstrap_replacement:
    :param run:
    :param fold:
    :param numTrain:
    :param bootstrap_ratio:
    :return:
    """
    seed = 2015 + 1000 * 3 + 10 * 3
    return bootstrap_data(seed, bootstrap_replacement, numTrain, bootstrap_ratio)


def bootstrap_run_fold(bootstrap_replacement, run, fold, numTrain, bootstrap_ratio):
    """
    对交叉验证数据进行自举法抽样
    :param bootstrap_replacement:
    :param numTrain:
    :param bootstrap_ratio:
    :return:
    """
    seed = 2015 + 1000 * run + 10 * fold
    return bootstrap_data(seed, bootstrap_replacement, numTrain, bootstrap_ratio)


def bootstrap_data(seed, bootstrap_replacement, numTrain, bootstrap_ratio):
    """
    使用指定的种子；从训练数据中抽取一定比例的数据，返回数据索引
    :param seed:
    :param bootstrap_replacement: 放回的抽样；不放回的抽样
    :param numTrain:
    :param bootstrap_ratio:
    :return:返回抽样索引，没抽中的索引
    """
    rng = np.random.RandomState(seed)
    if bootstrap_replacement:
        sampleSize = int(numTrain * bootstrap_ratio)
        # 每个元素在 [0-numTrain)之间 ，共 sampleSize个这样的随机数字组成的数组array
        index_base = rng.randint(numTrain, size=sampleSize)
        # 没有抽到的元素
        index_meta = [i for i in range(numTrain) if i not in index_base]
    else:
        # 不重复抽样
        randnum = rng.uniform(size=numTrain)
        index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
        index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]
    return index_base, index_meta


def try_divide(x, y, val=0.0):
    """
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val