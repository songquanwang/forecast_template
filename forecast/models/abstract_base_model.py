# -*- coding: utf-8 -*-
__author__ = 'songquanwang'

import abc
import csv

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

import xgboost as xgb
from hyperopt import STATUS_OK
from scipy.sparse import hstack
import forecast.conf.model_params_conf as model_param_conf
import forecast.utils.utils as utils
import forecast.conf.model_params_conf as config
import forecast.conf.model_library_config as model_conf
from forecast.interface.model_inter import ModelInter
import os


class AbstractBaseModel(ModelInter):
    __metaclass__ = abc.ABCMeta

    def __init__(self, param_space, info_folder, feat_folder, feat_name):
        self.param_space = param_space
        self.feat_folder = feat_folder
        self.info_folder = info_folder
        self.feat_name = feat_name
        self.all_matrix = dict()
        # self.run_fold_matrix = np.empty((config.n_runs, config.n_folds), dtype=object)
        # 创建一个dict矩阵用于存放run fold的各种集合
        self.run_fold_matrix = np.asarray([dict() for x in range(config.n_runs * config.n_folds)]).reshape(config.n_runs, config.n_folds)
        self.trial_counter = 0
        log_path = "%s/Log" % feat_folder
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = "%s/%s_hyperopt.log" % (log_path, feat_name)
        self.log_handler = open(log_file, 'wb')
        self.writer = csv.writer(self.log_handler)

    def init_all_path(self, matrix):
        feat_path = "%s/All/feat" % (self.feat_folder)
        info_path = "%s/All" % (self.info_folder)
        matrix['feat_train_path'] = "%s/train.feat" % feat_path
        matrix['feat_test_path'] = "%s/test.feat" % feat_path

        matrix['weight_train_path'] = "%s/train.feat.weight" % info_path

        matrix['info_train_path'] = "%s/train.info" % info_path
        matrix['info_test_path'] = "%s/test.info" % info_path

        matrix['cdf_test_path'] = "%s/test.cdf" % info_path

    def init_run_fold_path(self, run, fold, matrix):
        feat_path = "%s/Run%d/Fold%d/feat" % (self.feat_folder, run, fold)
        info_path = "%s/Run%d/Fold%d" % (self.info_folder, run, fold)
        matrix['feat_train_path'] = "%s/train.feat" % feat_path
        matrix['feat_valid_path'] = "%s/valid.feat" % feat_path

        matrix['weight_train_path'] = "%s/train.feat.weight" % info_path
        matrix['weight_valid_path'] = "%s/valid.feat.weight" % info_path

        matrix['info_train_path'] = "%s/train.info" % info_path
        matrix['info_valid_path'] = "%s/valid.info" % info_path

        matrix['cdf_valid_path'] = "%s/valid.cdf" % info_path

    def get_output_all_path(self, feat_folder, feat_name, kappa_cv_mean, kappa_cv_std):
        save_path = "%s/All/pred" % feat_folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        subm_path = "%s/Subm" % feat_folder
        if not os.path.exists(subm_path):
            os.makedirs(subm_path)
        raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, self.trial_counter)
        rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, self.trial_counter)
        # submission path (relevance as in [1,2,3,4]) 整合时候好像没用到
        subm_path = "%s/test.pred.%s_[Id@%d]_[Mean%.6f]_[Std%.6f].csv" % (subm_path, feat_name, self.trial_counter, kappa_cv_mean, kappa_cv_std)

        return raw_pred_test_path, rank_pred_test_path, subm_path

    def get_output_run_fold_path(self, feat_folder, feat_name, run, fold):
        save_path = "%s/Run%d/Fold%d/pred" % (feat_folder, run, fold)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, self.trial_counter)
        rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, self.trial_counter)

        return raw_pred_valid_path, rank_pred_valid_path

    def gen_set_obj_all(self, matrix):
        # init the path
        self.init_all_path(matrix)
        # feat
        X_train, labels_train = load_svmlight_file(matrix['feat_train_path'])
        X_test, labels_test = load_svmlight_file(matrix['feat_test_path'])
        # 延展array
        if X_test.shape[1] < X_train.shape[1]:
            X_test = hstack([X_test, np.zeros((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])
        elif X_test.shape[1] > X_train.shape[1]:
            X_train = hstack([X_train, np.zeros((X_train.shape[0], X_test.shape[1] - X_train.shape[1]))])
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()
        # 赋给成员变量
        matrix['X_train'], matrix['labels_train'], matrix['X_test'], matrix['labels_test'] = X_train, labels_train, X_test, labels_test
        # weight
        matrix['weight_train'] = np.loadtxt(matrix['weight_train_path'], dtype=float)
        # info
        matrix['info_train'] = pd.read_csv(matrix['info_train_path'])
        matrix['info_test'] = pd.read_csv(matrix['info_test_path'])
        # cdf
        matrix['cdf_test'] = np.loadtxt(matrix['cdf_test_path'], dtype=float)
        # number
        matrix['numTrain'] = matrix['info_train'].shape[0]
        matrix['numTest'] = matrix['info_test'].shape[0]
        # id
        matrix['id_test'] = matrix['info_test']["id"]

        return matrix

    def gen_set_obj_run_fold(self, run, fold, matrix):
        """
        每个run 每个fold 生成
        :param run:
        :param fold:
        :return:
        """
        # init the path
        self.init_run_fold_path(run, fold, matrix)
        # feat
        X_train, labels_train = load_svmlight_file(matrix['feat_train_path'])
        X_valid, labels_valid = load_svmlight_file(matrix['feat_valid_path'])
        # 延展array
        if X_valid.shape[1] < X_train.shape[1]:
            X_valid = hstack([X_valid, np.zeros((X_valid.shape[0], X_train.shape[1] - X_valid.shape[1]))])
        elif X_valid.shape[1] > X_train.shape[1]:
            X_train = hstack([X_train, np.zeros((X_train.shape[0], X_valid.shape[1] - X_train.shape[1]))])
        X_train = X_train.tocsr()
        X_valid = X_valid.tocsr()
        # 赋给成员变量
        matrix['X_train'], matrix['labels_train'], matrix['X_valid'], matrix['labels_valid'] = X_train, labels_train, X_valid, labels_valid
        # weight
        matrix['weight_train'] = np.loadtxt(matrix['weight_train_path'], dtype=float)
        matrix['weight_valid'] = np.loadtxt(matrix['weight_valid_path'], dtype=float)
        # info
        matrix['info_train'] = pd.read_csv(matrix['info_train_path'])
        matrix['info_valid'] = pd.read_csv(matrix['info_valid_path'])
        # cdf
        matrix['cdf_valid'] = np.loadtxt(matrix['cdf_valid_path'], dtype=float)
        # number
        matrix['numTrain'] = matrix['info_train'].shape[0]
        matrix['numValid'] = matrix['info_valid'].shape[0]
        # Y valid
        matrix['Y_valid'] = matrix['info_valid']["median_relevance"]

        return matrix

    def out_put_all(self, feat_folder, feat_name, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank):
        # write
        output = pd.DataFrame({"id": self.all_matrix['id_test'], "prediction": pred_raw})
        output.to_csv(self.all_matrix['raw_pred_test_path'], index=False)

        # write
        output = pd.DataFrame({"id": self.all_matrix['id_test'], "prediction": pred_rank})
        output.to_csv(self.all_matrix['rank_pred_test_path'], index=False)

        # write score pred--原来代码有错：应该是pred_raw 因为pred_raw是多次装袋后平均预测值，不应该是其中一次装袋的预测值
        pred_score = utils.getScore(pred_raw, self.all_matrix['cdf_test'])
        output = pd.DataFrame({"id": self.all_matrix['id_test'], "prediction": pred_score})
        output.to_csv(self.all_matrix['subm_path'], index=False)

    def out_put_run_fold(self, run, fold, feat_folder, feat_name, X_train, Y_valid, pred_raw, pred_rank, kappa_valid):
        """

        :param run:
        :param fold:
        :param bagging:
        :param feat_name:
        :param kappa_valid:
        :param X_train:
        :param Y_valid:
        :param pred_raw:
        :param pred_rank:
        :return:
        """
        matrix = self.run_fold_matrix[run - 1, fold - 1]
        # save this prediction
        dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_raw})
        dfPred.to_csv(matrix['raw_pred_valid_path'], index=False, header=True, columns=["target", "prediction"])
        # save this prediction
        dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_rank})
        dfPred.to_csv(matrix['rank_pred_valid_path'], index=False, header=True, columns=["target", "prediction"])

    def gen_bagging(self, param, set_obj, all):
        """
        分袋整合预测结果
        :param set_obj:
        :param all:
        :return:
        """
        for n in range(model_param_conf.bagging_size):
            # 对数据进行自举法抽样；因为ratio=1 且bootstrap_replacement=false 说明没有用到，就使用的是全量数据
            index_base, index_meta = utils.bootstrap_all(model_param_conf.bootstrap_replacement, set_obj['numTrain'], model_param_conf.bootstrap_ratio)
            set_obj['index_base'] = index_base
            set_obj['dtrain'] = xgb.DMatrix(set_obj['X_train'][index_base], label=set_obj['labels_train'][index_base], weight=set_obj['weight_train'][index_base])
            if all:
                preds_bagging = np.zeros((set_obj['numTest'], model_param_conf.bagging_size), dtype=float)
                set_obj['dtest'] = xgb.DMatrix(set_obj['X_test'], label=set_obj['labels_test'])
                # watchlist
                set_obj['watchlist'] = []
                if model_param_conf.verbose_level >= 2:
                    set_obj['watchlist'] = [(set_obj['dtrain'], 'train')]
                    # 调用 每个子类的train_predict方法，多态
                pred = self.train_predict(param, set_obj, all)
                pred_test = pred
                preds_bagging[:, n] = pred_test
            else:
                preds_bagging = np.zeros((set_obj['numValid'], model_param_conf.bagging_size), dtype=float)
                set_obj['dvalid'] = xgb.DMatrix(set_obj['X_valid'], label=set_obj['labels_valid'])
                # watchlist
                set_obj['watchlist'] = []
                if model_param_conf.verbose_level >= 2:
                    set_obj['watchlist'] = [(set_obj['dtrain'], 'train'), (set_obj['dvalid_base'], 'valid')]
                # 调用 每个子类的train_predict方法，多态
                pred = self.train_predict(param, set_obj, all)
                pred_valid = pred
                preds_bagging[:, n] = pred_valid
                # 每次会把当前bagging的结果累计进来 求均值
                pred_raw = np.mean(preds_bagging[:, :(n + 1)], axis=1)
                # 为什么需要两次argsort？
                pred_rank = pred_raw.argsort().argsort()
                pred_score, cutoff = utils.getScore(pred_rank, set_obj['cdf_valid'], valid=True)
                kappa_valid = utils.quadratic_weighted_kappa(pred_score, set_obj['Y_valid'])

        if all:
            pred_raw = np.mean(preds_bagging, axis=1)
            pred_rank = pred_raw.argsort().argsort()
            return pred_raw, pred_rank
        else:
            return pred_raw, pred_rank, kappa_valid

    def hyperopt_obj(self, param, feat_folder, feat_name):
        """
        最优化方法 hyperopt_obj
        :param feat_folder:
        :param feat_name:
        :param trial_counter:
        :return:
        """
        self.trial_counter += 1
        # 定义kappa交叉验证结构
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(1, config.n_runs + 1):
            for fold in range(1, config.n_folds + 1):
                # 生成 run_fold_set_obj
                set_obj = self.gen_set_obj_run_fold(run, fold, self.run_fold_matrix[run - 1, fold - 1])
                # 获取输出目录赋值给set_obj
                set_obj['raw_pred_valid_path'], set_obj['rank_pred_valid_path'] = self.get_output_run_fold_path(feat_folder, feat_name, run, fold)
                # bagging结果
                pred_raw, pred_rank, kappa_valid = self.gen_bagging(param, set_obj, all=False)
                # 输出文件
                kappa_cv[run - 1, fold - 1] = kappa_valid
                # 生成没run fold的结果
                self.out_put_run_fold(run, fold, feat_folder, feat_name, set_obj['X_train'], set_obj['Y_valid'], pred_raw, pred_rank, kappa_valid)
        # kappa_cv run*fold*bagging_size 均值和方差
        kappa_cv_mean, kappa_cv_std = np.mean(kappa_cv), np.std(kappa_cv)
        if model_param_conf.verbose_level >= 1:
            print(" Mean: %.6f" % kappa_cv_mean)
            print(" Std: %.6f" % kappa_cv_std)
        # all result
        all_set_obj = self.gen_set_obj_all(self.all_matrix)
        # 输出路径赋值给all_matrix
        self.all_matrix['raw_pred_test_path'], self.all_matrix['rank_pred_test_path'], self.all_matrix['subm_path'] = self.get_output_all_path(feat_folder, feat_name, kappa_cv_mean, kappa_cv_std)
        pred_raw, pred_rank = self.gen_bagging(param, all_set_obj, all=True)
        # 生成提交结果
        self.out_put_all(feat_folder, feat_name, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank)
        # 记录参数文件
        self.log_param(param, feat_name, kappa_cv_mean, kappa_cv_std)
        # 根据交叉验证的平均值作为模型好坏标准
        return {'loss': -kappa_cv_mean, 'attachments': {'std': kappa_cv_std}, 'status': STATUS_OK}

    def log_header(self):
        """
        log 记录文件头部
        :return:
        """
        # 记录日志 到output/***_hyperopt.log
        # 每行日志都包含 'trial_counter', 'kappa_mean', 'kappa_std' 三个字段 + 模型参数
        headers = ['trial_counter', 'kappa_mean', 'kappa_std']
        for k, v in sorted(self.param_space.items()):
            headers.append(k)
        self.writer.writerow(headers)
        self.log_handler.flush()

    def log_param(self, param, feat_name, kappa_cv_mean, kappa_cv_std):
        """
        记录参数文件
        :param param:
        :param feat_name:
        :param kappa_cv_mean:
        :param kappa_cv_std:
        :return:
        """
        # convert integer feat
        for f in model_conf.int_feat:
            if param.has_key(f):
                param[f] = int(param[f])

        print("------------------------------------------------------------")
        print "Trial %d" % self.trial_counter

        print(" Model")
        print(" %s" % feat_name)
        print(" Param")
        for k, v in sorted(param.items()):
            print(" %s: %s" % (k, v))
        print(" Result")
        print("                    Run      Fold      Bag      Kappa      Shape")
        # log
        var_to_log = [
            "%d" % self.trial_counter,
            "%.6f" % kappa_cv_mean,
            "%.6f" % kappa_cv_std
        ]
        # 日志中输出参数值 次数 均值 标准差 值1 值2 ...值N
        for k, v in sorted(param.items()):
            var_to_log.append("%s" % v)
        self.writer.writerow(var_to_log)
        self.log_handler.flush()
