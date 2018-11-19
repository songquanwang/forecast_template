# coding:utf-8
__author__ = 'songquanwang'

import abc


class ModelInter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_all_path(self, matrix):
        return

    @abc.abstractmethod
    def init_run_fold_path(self, run, fold, matrix):
        return

    @abc.abstractmethod
    def get_output_all_path(self, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std):
        return

    @abc.abstractmethod
    def get_output_run_fold_path(self, feat_name, trial_counter, run, fold):
        return

    @abc.abstractmethod
    def gen_set_obj_all(self, matrix):
        # init the path
        return

    @abc.abstractmethod
    def gen_set_obj_run_fold(self, run, fold, matrix):
        """
        每个run 每个fold 生成
        :param run:
        :param fold:
        :return:
        """
        return

    @abc.abstractmethod
    def out_put_all(self, feat_name, trial_counter, kappa_cv_mean, kappa_cv_std, pred_raw, pred_rank):
        return

    @abc.abstractmethod
    def out_put_run_fold(self, run, fold, feat_name, trial_counter, X_train, Y_valid, pred_raw, pred_rank, kappa_valid):
        """

        :param run:
        :param fold:
        :param bagging:
        :param feat_name:
        :param trial_counter:
        :param kappa_valid:
        :param X_train:
        :param Y_valid:
        :param pred_raw:
        :param pred_rank:
        :return:
        """
        return

    def gen_bagging(self, param, set_obj, all):
        """
        分袋整合预测结果
        :param set_obj:
        :param all:
        :return:
        """
        return

    @abc.abstractmethod
    def hyperopt_obj(self, param, feat_folder, feat_name, trial_counter):
        """
        最优化方法 hyperopt_obj
        :param feat_folder:
        :param feat_name:
        :param trial_counter:
        :return:
        """
        return

    @abc.abstractmethod
    def train_predict(self, matrix, all=False):
        """
        所有子类模型都需要实现这个方法
        :param matrix:
        :param all:
        :return:
        """
        return
