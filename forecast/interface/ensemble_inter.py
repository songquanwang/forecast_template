# coding:utf-8

"""
__file__

    ensemble_inter.py

__description__

    This file contains ensemble selection module.

__author__

    songquanwang

"""

import abc


class EnsembleInter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def ensemble_bagging_models_prediction(self, best_bagged_model_list, best_bagged_model_weight, cdf, cutoff=None):
        """
        按照bagging、model_list 集成预测结果；根据交叉验证选取的最佳模型，集成All预测结果
        :param best_bagged_model_list:
        :param best_bagged_model_weight:
        :param cdf:
        :param cutoff:
        :return:
        """
        return

    @abc.abstractmethod
    def gen_best_model_and_weight(self, this_sorted_models, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix, w_min, w_max, hypteropt_max_evals, best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens):
        """
        一组模型，循环遍历，只集成比当前模型好的
        :param this_sorted_models:
        :param w_min:
        :param w_max:
        :param hypteropt_max_evals:
        :param w_ens: topk集成结果权重
        :param p_ens_list_valid_topk:topk集成结果
        :param best_model_list:引用 ；在topk基础上添加
        :param best_model_weight:引用；在topk基础上添加
        :return:
        """
        return

    @abc.abstractmethod
    def ensemble_model_list_pedicts(self, feat_folder, cdf, cdf_test, subm_prefix, hypteropt_max_evals=10, w_min=-1., w_max=1., bagging_replacement=False, bagging_fraction=0.5, bagging_size=10,
                                    init_top_k=5,
                                    prunning_fraction=0.2):
        """

        :param feat_folder:
        :param model_folder:
        :param model_list:
        :param cdf:
        :param cdf_test:
        :param subm_prefix:
        :param hypteropt_max_evals:
        :param w_min:
        :param w_max:
        :param bagging_replacement:
        :param bagging_fraction:
        :param bagging_size:
        :param init_top_k:
        :param prunning_fraction:
        :return:
        """
        return
