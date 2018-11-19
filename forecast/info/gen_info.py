# coding:utf-8
"""
__file__

    gen_info.py

__description__

    This file generates the following info for each run and fold, and for the entire training and testing set.

        1. training and validation/testing data

        2. sample weight

        3. cdf of the median_relevance
        
        4. the group info for pairwise ranking in XGBoost

__author__

    songquanwang

"""

import os
import pickle

import numpy as np
import pandas as pd

import forecast.conf.model_params_conf as  config


def gen_run_fold_info(run, fold, dfTrain, trainInd, validInd, dfTrain_original, Y):
    """
    注意，没有 dfTest，dfTest_original 。 run fold是把训练数据分成 训练和验证两部分
    :param feat_path_name:
    :param run:
    :param fold:
    :param dfTrain:
    :param trainInd:
    :param validInd:
    :param dfTrain_original:
    :return:
    """

    print("Run: %d, Fold: %d" % (run + 1, fold + 1))
    path = "%s/Run%d/Fold%d" % (config.solution_info, run + 1, fold + 1)
    if not os.path.exists(path):
        os.makedirs(path)
    ##########################
    ## get and dump weights ##
    ##########################
    raise_to = 0.5
    var = dfTrain["relevance_variance"].values
    # 最大标准差
    max_var = np.max(var[trainInd] ** raise_to)
    # [1+（最大标准差-标准差数组）/最大标准差]/2   --->标准差越大，权重越小  （0.5-1)
    weight = (1 + np.power(((max_var - var ** raise_to) / max_var), 1)) / 2.
    # weight = (max_var - var**raise_to) / max_var
    np.savetxt("%s/train.feat.weight" % path, weight[trainInd], fmt="%.6f")
    np.savetxt("%s/valid.feat.weight" % path, weight[validInd], fmt="%.6f")

    #############################
    ## get and dump group info ##
    #############################
    np.savetxt("%s/train.feat.group" % path, [len(trainInd)], fmt="%d")
    np.savetxt("%s/valid.feat.group" % path, [len(validInd)], fmt="%d")

    ######################
    ## get and dump cdf ##
    ######################
    hist = np.bincount(Y[trainInd])
    overall_cdf_valid = np.cumsum(hist) / float(sum(hist))
    np.savetxt("%s/valid.cdf" % path, overall_cdf_valid)

    #############################
    ## dump all the other info ##
    #############################
    dfTrain_original.iloc[trainInd].to_csv("%s/train.info" % path, index=False, header=True)
    dfTrain_original.iloc[validInd].to_csv("%s/valid.info" % path, index=False, header=True)


def gen_all_info(dfTrain, dfTest, dfTrain_original, dfTest_original, Y):
    """
    没有test.feat.weight
    :param feat_path_name:
    :param dfTrain:
    :param dfTest:
    :param dfTrain_original:
    :param dfTest_original:
    :param Y:
    :return:
    """
    print("For training and testing...")
    path = "%s/All" % config.solution_info
    raise_to = 0.5
    var = dfTrain["relevance_variance"].values
    if not os.path.exists(path):
        os.makedirs(path)
    ## weight
    max_var = np.max(var ** raise_to)
    weight = (1 + np.power(((max_var - var ** raise_to) / max_var), 1)) / 2.
    np.savetxt("%s/train.feat.weight" % path, weight, fmt="%.6f")

    ## group
    np.savetxt("%s/train.feat.group" % path, [dfTrain.shape[0]], fmt="%d")
    np.savetxt("%s/test.feat.group" % path, [dfTest.shape[0]], fmt="%d")
    ## cdf
    hist_full = np.bincount(Y)
    print (hist_full) / float(sum(hist_full))
    overall_cdf_full = np.cumsum(hist_full) / float(sum(hist_full))
    np.savetxt("%s/test.cdf" % path, overall_cdf_full)
    ## info
    dfTrain_original.to_csv("%s/train.info" % path, index=False, header=True)
    dfTest_original.to_csv("%s/test.info" % path, index=False, header=True)


def gen_info():
    """
    生成模型所用的 weight、cdf、info、group等文件
    :param feat_path_name:
    :return:
    """
    # 打开预处理后的数据
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)
    # 打开原始数据
    dfTrain_original = pd.read_csv(config.original_train_data_path).fillna("")
    dfTest_original = pd.read_csv(config.original_test_data_path).fillna("")
    # 为test插入假的label（test没有label） 相关性全置1；方差全置0
    dfTest_original["median_relevance"] = np.ones((dfTest_original.shape[0]))
    dfTest_original["relevance_variance"] = np.zeros((dfTest_original.shape[0]))
    # change it to zero-based for classification
    Y = dfTrain_original["median_relevance"].values - 1

    # load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, config.stratified_label), "rb") as f:
        skf = pickle.load(f)

    print("Generate info...")
    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            gen_run_fold_info( run, fold, dfTrain, trainInd, validInd, dfTrain_original, Y)

    print("Done.")

    print("For training and testing...")
    # 生成all
    gen_all_info( dfTrain, dfTest, dfTrain_original, dfTest_original, Y)
    print("All Done.")



