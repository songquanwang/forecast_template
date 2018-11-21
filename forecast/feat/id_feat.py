# coding:utf-8
"""
__file__

    genFeat_id_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. one-hot encoding of query ids (qid)

__author__

    songquanwang

"""

import _pickle as pickle

from sklearn.preprocessing import LabelBinarizer

import forecast.conf.model_params_conf as config

import abc
from  forecast.feat.abstract_base_feat import AbstractBaseFeat


class IdFeat(AbstractBaseFeat):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def gen_feat(path, dfTrain, dfTest, mode, feat_names, lb):
        for feat_name in feat_names:
            # 返回 numpy array or CSR matrix of shape [n_samples, n_classes]
            X_train = lb.fit_transform(dfTrain[feat_name])
            X_test = lb.transform(dfTest[feat_name])
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                pickle.dump(X_train, f, -1)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
                pickle.dump(X_test, f, -1)

    def gen_feat_cv(self):
        """
        入口函数
        :return:
        """
        feat_names = ["qid"]
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = pickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            dfTest = pickle.load(f)
        ## load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, config.stratified_label), "rb") as f:
            skf = pickle.load(f)

        lb = LabelBinarizer(sparse_output=True)

        print("==================================================")
        print("Generate id features...")

        print("For cross-validation...")
        for run in range(config.n_runs):
            # use 33% for training and 67 % for validation so we switch trainInd and validInd
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                path = "%s/Run%d/Fold%d" % (config.solution_feat_base, run + 1, fold + 1)
                # 生成 run fold
                X_train_train = dfTrain.iloc[trainInd]
                X_train_valid = dfTrain.iloc[validInd]
                self.gen_feat(path, X_train_train, X_train_valid, "valid", feat_names, lb)

        print("Done.")

        print("For training and testing...")
        path = "%s/All" % config.solution_feat_base
        self.gen_feat(path, dfTrain, dfTest, "test", feat_names, lb)
        print("Done.")

        # 保存特征文件
        feat_name_file = "%s/id.feat_name" % (config.solution_feat_combined)
        self.dump_feat_name(feat_names, feat_name_file)

        print("All Done.")
