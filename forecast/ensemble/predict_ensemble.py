# coding:utf-8
"""
__file__

    predict_ensemble.py

__description__

    This file contains ensemble selection module.

__author__

    songquanwang

"""

import abc
import os

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from forecast.utils.utils import getScore, getTestScore
from forecast.utils.ml_metrics import quadratic_weighted_kappa
import forecast.conf.model_params_conf as config
import forecast.conf.model_library_config as model_library_config
import forecast.utils.utils as utils
from  forecast.interface.ensemble_inter import EnsembleInter


class PredictEnsemble(EnsembleInter):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_folder, subm_folder):
        self.model_folder = model_folder
        self.subm_folder = subm_folder
        self.model_list = []
        self.model2idx = dict()

        # 每个model的kappa值
        self.max_num_valid = 12000
        id_sizes = config.ensemble_model_top__k * np.ones(len(model_library_config.feat_names), dtype=int)

        # 读取各个模型hyperopt文件
        for feat_name, id_size in zip(model_library_config.feat_names, id_sizes):
            # 获取每个算法的前十个模型；根据kappa_mean排序
            log_file = "%s/Log/%s_hyperopt.log" % (self.model_folder, feat_name)
            try:
                # 读取模型平均参数
                dfLog = pd.read_csv(log_file)
                # kappa mean降序排列
                dfLog.sort_values("kappa_mean", ascending=False, inplace=True)
                ind = np.min([id_size, dfLog.shape[0]])
                # 取出前十行 也就是当前模型的前十名尝试累加到数组
                ids = dfLog.iloc[:ind]["trial_counter"]
                # print dfLog[:ind]
                self.model_list += ["%s_[Id@%d]" % (feat_name, id-1) for id in ids]
            except:
                pass
        # 初始化
        for i, model in enumerate(self.model_list):
            self.model2idx[model] = i

        # 不存在则创建目录
        if not os.path.exists(self.subm_folder):
            os.makedirs(self.subm_folder)

    def ensemble_bagging_models_prediction(self, best_bagged_model_list, best_bagged_model_weight, cdf, cutoff=None):
        """
        按照bagging、model_list 集成预测结果；根据交叉验证选取的最佳模型，集成All预测结果
        :param best_bagged_model_list:
        :param best_bagged_model_weight:
        :param cdf:
        :param cutoff:
        :return:
        """
        bagging_size = len(best_bagged_model_list)
        # 多次分袋
        for bagging_iter in range(bagging_size):
            # 初始化累计权重
            w_ens = 0
            iter = 0
            # 多个模型集成结果(All预测结果)
            for model, w in zip(best_bagged_model_list[bagging_iter], best_bagged_model_weight[bagging_iter]):
                iter += 1
                pred_file = "%s/All/pred/test.pred.%s.csv" % (self.model_folder, model)
                # 获取当前模型预测值
                this_p_valid = pd.read_csv(pred_file, dtype=float)["prediction"].values
                this_w = w
                if iter == 1:
                    # 初始化整合预测值是0
                    p_ens_valid = np.zeros((this_p_valid.shape[0]), dtype=float)
                    id_test = pd.read_csv(pred_file, dtype=float)["id"].values
                    id_test = np.asarray(id_test, dtype=int)
                # 按照归一化权重 线性组合
                p_ens_valid = (w_ens * p_ens_valid + this_w * this_p_valid) / (w_ens + this_w)
                # 累计权重
                w_ens += this_w
            # 多个bagging进行集成，每个bagging的权重都相同
            if bagging_iter == 0:
                p_ens_valid_bag = p_ens_valid
            else:
                # 每次bagging的权重都是1，同等权重
                p_ens_valid_bag = (bagging_iter * p_ens_valid_bag + p_ens_valid) / (bagging_iter + 1.)
        # 根据cdf对排序后的预测结果进行映射成1-4
        if cutoff is None:
            p_ens_score = getScore(p_ens_valid_bag, cdf)
        else:
            # 使用相近取整的方式得出预测结果
            p_ens_score = getTestScore(p_ens_valid_bag, cutoff)
        # 输出集成后的结果
        output = pd.DataFrame({"id": id_test, "prediction": p_ens_score})
        return output

    def init_model_metrics_by_run_fold(self, feat_folder, cdf):
        """
         为每个交叉验证数据按照 run-fold生成一系列指标
         初始化实例变量,供后续方法使用
         kappa_list      ：每个模型的平均kappa值
         num_valid_matrix:每个run-fold 的预测结果行数
         y_list_valid    :每个run-fold 的真实label
         cdf_list_valid  ：每个run-fold 的cdf
         kappa_cv        ：每个run-fold 的kappa cv
         pred_list_valid :每个run-fold 的真实预测值

        :param feat_folder:
        :param cdf:
        :return:
        """
        kappa_list = dict()
        # 模型-run-fold-行 交叉验证-valid数据集预测结果
        pred_list_valid = np.zeros((len(self.model_list), config.n_runs, config.n_folds, self.max_num_valid), dtype=float)
        # run-fold-行      交叉验证-valid数据集真实label
        y_list_valid = np.zeros((config.n_runs, config.n_folds, self.max_num_valid), dtype=float)
        # run-fold-4类别   交叉验证-valid数据集预测结果cdf
        cdf_list_valid = np.zeros((config.n_runs, config.n_folds, config.num_of_class), dtype=float)
        # run-fold valid   交叉验证-valid数据集预测结果行数
        num_valid_matrix = np.zeros((config.n_runs, config.n_folds), dtype=int)
        print("Load model...")
        for i, model in enumerate(self.model_list):
            print("model: %s" % model)
            kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    path = "%s/Run%d/Fold%d/pred" % (self.model_folder, run + 1, fold + 1)
                    pred_file = "%s/valid.pred.%s.csv" % (path, model)
                    cdf_file = "%s/Run%d/Fold%d/valid.cdf" % (config.solution_info, run + 1, fold + 1)
                    this_p_valid = pd.read_csv(pred_file, dtype=float)
                    # 这些指标只需要执行一次就行了，每个模型都一样
                    if i == 0:
                        # 记录run-fold的行数
                        num_valid_matrix[run][fold] = this_p_valid.shape[0]
                        # 记录run-fold的真实值
                        y_list_valid[run, fold, :num_valid_matrix[run][fold]] = this_p_valid["target"].values
                        # load cdf
                        if cdf == None:
                            cdf_list_valid[run, fold, :] = np.loadtxt(cdf_file, dtype=float)
                        else:
                            cdf_list_valid[run, fold, :] = cdf
                        score = getScore(this_p_valid["prediction"].values, cdf_list_valid[run, fold, :])
                        kappa_cv[run][fold] = quadratic_weighted_kappa(score, y_list_valid[run, fold, :num_valid_matrix[run][fold]])
                    # 记录model-run-fold的预测值数组
                    pred_list_valid[self.model2idx[model], run, fold, :this_p_valid.shape[0]] = this_p_valid["prediction"].values
            print("kappa: %.6f" % np.mean(kappa_cv))
            # 算出每个模型的平均kappa_cv
            kappa_list[model] = np.mean(kappa_cv)

        return kappa_list, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix

    def init_topk_best_model(self, init_top_k, this_sorted_models, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix):
        """
        选择前五个模型 返回整合后的预测值；前五个模型名字；前五个模型的权重(全是1,相当于取平均值)
        读取实例变量：
        pred_list_valid
        num_valid_matrix
        model2idx
        cdf_list_valid
        y_list_valid
        :param init_top_k:
        :param this_sorted_models:
        :return:best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens
        """
        best_model_list = []
        best_model_weight = []
        p_ens_list_valid_topk = np.zeros((config.n_runs, config.n_folds, self.max_num_valid), dtype=float)
        w_ens, this_w = 0, 1.0
        cnt = 0
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for model, kappa in this_sorted_models[0:init_top_k]:
            print("add the following model to the ensembles ")
            print("model: %s" % model)
            print("kappa: %.6f" % kappa)
            # 指定模型的预测结果
            this_p_list_valid = pred_list_valid[self.model2idx[model]]
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    num_valid = num_valid_matrix[run][fold]
                    # 多个模型预测值线性组合
                    p_ens_list_valid_topk[run, fold, :num_valid] = (w_ens * p_ens_list_valid_topk[run, fold, :num_valid] + this_w * this_p_list_valid[run, fold, :num_valid]) / (w_ens + this_w)
                    # 在最后一个model，生成一些指标
                    if cnt == init_top_k - 1:
                        cdf = cdf_list_valid[run, fold, :]
                        true_label = y_list_valid[run, fold, :num_valid]
                        score = getScore(p_ens_list_valid_topk[run, fold, :num_valid], cdf)
                        kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
            best_model_list.append(model)
            best_model_weight.append(this_w)
            w_ens += this_w
            cnt += 1
            print("Init kappa: %.6f (%.6f)" % (np.mean(kappa_cv), np.std(kappa_cv)))
        return best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens

    def ensemble_selection_obj(self, param, p1_list, weight1, p2_list, y_list_valid, cdf_list_valid, num_valid_matrix):
        """
        优化param中的weight_current_model参数，使其平均kappa_cv_mean 最大
        :param param:
        :param p1_list: 集成前五个模型(也就是对前五个模型求平均值的结果)
        :param weight1: 1
        :param p2_list: 当前模型预测结果
        :return:
        """
        weight_current_model = param['weight_current_model']
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                num_valid = num_valid_matrix[run][fold]
                p1 = p1_list[run, fold, :num_valid]
                p2 = p2_list[run, fold, :num_valid]
                true_label = y_list_valid[run, fold, :num_valid]
                cdf = cdf_list_valid[run, fold, :]
                p_ens = (weight1 * p1 + weight_current_model * p2) / (weight1 + weight_current_model)
                p_ens_score = getScore(p_ens, cdf)
                kappa_cv[run][fold] = quadratic_weighted_kappa(p_ens_score, true_label)
        kappa_cv_mean = np.mean(kappa_cv)
        return {'loss': -kappa_cv_mean, 'status': STATUS_OK}

    def find_best_model(self, this_sorted_models, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix, w_ens, w_min, w_max, best_kappa, hypteropt_max_evals, p_ens_list_valid_topk):
        """
        从模型集合中找出一个最佳模型，最佳系数，与topK集成结果进行线性组合
        寻找最佳模型、权重、kappa值 从this_sorted_models找到一个最佳模型
        :param this_sorted_models:
        :param w_ens:
        :param w_min:
        :param w_max:
        :param hypteropt_max_evals:
        :param p_ens_list_valid_topk:
        :return:
        """
        best_model = None
        best_weight = 0
        for model, kappa in this_sorted_models:
            # 当前模型预测值
            this_p_list_valid = pred_list_valid[self.model2idx[model]]
            # hyperopt 找当前模型最优权重
            trials = Trials()
            # 不同模型的权重
            param_space = {
                'weight_current_model': hp.uniform('weight_current_model', w_min, w_max)
            }
            # topk权重是1 找另一个最佳权重
            obj = lambda param: self.ensemble_selection_obj( param, p_ens_list_valid_topk, 1., this_p_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix)
            best_params = fmin(obj, param_space, algo=tpe.suggest, trials=trials, max_evals=hypteropt_max_evals)
            # 返回当前模型权重
            this_w = best_params['weight_current_model']
            # 按比例缩放当前权重 1 this_w --- w_ens this_w * w_ens
            this_w *= w_ens
            # 当前kappa cv
            kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    num_valid = num_valid_matrix[run][fold]
                    # topk预测
                    p1 = p_ens_list_valid_topk[run, fold, :num_valid]
                    # 当前预测
                    p2 = this_p_list_valid[run, fold, :num_valid]
                    # 真实label
                    true_label = y_list_valid[run, fold, :num_valid]
                    cdf = cdf_list_valid[run, fold, :]
                    # 集成后的结果
                    p_ens = (w_ens * p1 + this_w * p2) / (w_ens + this_w)
                    score = getScore(p_ens, cdf)
                    # 集成后kappa值
                    kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
            # 集成后平均kappa cv 由于现在
            if np.mean(kappa_cv) > best_kappa:
                best_kappa, best_model, best_weight = np.mean(kappa_cv), model, this_w
        return best_kappa, best_model, best_weight

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
        iter = 0
        best_kappa = 0
        best_model = None
        while True:
            iter += 1
            # 根据 当前best_kappa 找到一个比他平均kappa更好的模型;每次w_ens,best_kappa、p_ens_list_valid_topk 变化；应该会重复找到同一个模型？？
            best_kappa, best_model, best_weight = self.find_best_model(this_sorted_models, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix, w_ens, w_min, w_max, best_kappa, hypteropt_max_evals,
                                                                       p_ens_list_valid_topk)
            # 这次迭代找不到best_kappa更好的模型则终止
            if best_model == None:
                break
            print("Iter: %d" % iter)
            print("    model: %s" % best_model)
            print("    weight: %s" % best_weight)
            print("    kappa: %.6f" % best_kappa)
            best_model_list.append(best_model)
            best_model_weight.append(best_weight)
            # 最优模型
            this_p_list_valid = pred_list_valid[self.model2idx[best_model]]
            for run in range(config.n_runs):
                for fold in range(config.n_folds):
                    num_valid = num_valid_matrix[run][fold]
                    # 历史累计集成结果和当前（比历史上最佳好的模型）的集成结果
                    p_ens_list_valid_topk[run, fold, :num_valid] = (w_ens * p_ens_list_valid_topk[run, fold, :num_valid] + best_weight * this_p_list_valid[run, fold, :num_valid]) / (w_ens + best_weight)
            best_model = None
            w_ens += best_weight
        return best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens

    def gen_kappa_cv(self, bagging_iter, y_list_valid, cdf_list_valid, num_valid_matrix, p_ens_list_valid_topk, p_ens_list_valid):
        """
        多次bagging 的结果平均值
        :param bagging_iter:第几次bagging，有几次，权重是几
        :param p_ens_list_valid: 多次执行有状态
        :param p_ens_list_valid_topk:
        :return:
        """
        kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
        cutoff = np.zeros((3), dtype=float)
        for run in range(config.n_runs):
            for fold in range(config.n_folds):
                numValid = num_valid_matrix[run][fold]
                true_label = y_list_valid[run, fold, :numValid]
                cdf = cdf_list_valid[run, fold, :]
                # 每次bagging的结果平均
                p_ens_list_valid[run, fold, :numValid] = (bagging_iter * p_ens_list_valid[run, fold, :numValid] + p_ens_list_valid_topk[run, fold, :numValid]) / (bagging_iter + 1.)
                score, cutoff_tmp = getScore(p_ens_list_valid[run, fold, :numValid], cdf, "valid")
                kappa_cv[run][fold] = quadratic_weighted_kappa(score, true_label)
                cutoff += cutoff_tmp
        cutoff /= float(config.n_runs * config.n_folds)
        # 没搞懂？
        cutoff *= (22513 / ((2. / 3) * 10158))
        print("Bag %d, kappa: %.6f (%.6f)" % (bagging_iter + 1, np.mean(kappa_cv), np.std(kappa_cv)))
        return kappa_cv, cutoff, p_ens_list_valid

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

        print("Load model...")
        # 读取文件并初始化交叉验证相关集合
        kappa_list, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix = self.init_model_metrics_by_run_fold(feat_folder, cdf)

        # 按照 kappa_cv值 排序-置逆；大的靠前 ('model_id',cv)
        sorted_models = sorted(kappa_list.items(), key=lambda x: x[1])[::-1]

        # greedy ensemble
        print("============================================================")
        print("Perform ensemble selection...")
        best_bagged_model_list = [[]] * bagging_size
        best_bagged_model_weight = [[]] * bagging_size
        num_model = len(self.model_list)
        # run-fold-行      交叉验证-valid数据集整合后预测结果
        p_ens_list_valid = np.zeros((config.n_runs, config.n_folds, self.max_num_valid), dtype=float)
        # print bagging_size
        for bagging_iter in range(bagging_size):
            # 抽取一部分模型最为一次分袋
            index_base, index_meta = utils.bootstrap_data(2015 + 100 * bagging_iter, bagging_replacement, num_model, bagging_fraction)
            this_sorted_models = [sorted_models[i] for i in sorted(index_base)]

            # 用topk个模型做初始最佳模型，每个模型权值为1，valid_topk为topk个模型平均值
            best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens = self.init_topk_best_model(init_top_k, this_sorted_models, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix)

            # 获取多个最佳模型和权重
            best_model_list, best_model_weight, p_ens_list_valid_topk, w_ens = self.gen_best_model_and_weight(this_sorted_models, pred_list_valid, y_list_valid, cdf_list_valid, num_valid_matrix, w_min, w_max, hypteropt_max_evals, best_model_list,
                                                                                                              best_model_weight, p_ens_list_valid_topk, w_ens)
            # 对集成的最佳结果 --再多次bagging求平均
            kappa_cv, cutoff, p_ens_list_valid = self.gen_kappa_cv(bagging_iter, y_list_valid, cdf_list_valid, num_valid_matrix, p_ens_list_valid_topk, p_ens_list_valid)

            best_kappa_mean = np.mean(kappa_cv)
            best_kappa_std = np.std(kappa_cv)
            best_bagged_model_list[bagging_iter] = best_model_list
            best_bagged_model_weight[bagging_iter] = best_model_weight

            # save the current prediction 0-bagging个预测结果
            output = self.ensemble_bagging_models_prediction( best_bagged_model_list[:(bagging_iter + 1)], best_bagged_model_weight[:(bagging_iter + 1)], cdf_test)
            sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cdf.csv" % (subm_prefix, init_top_k, bagging_iter + 1, bagging_fraction, best_kappa_mean, best_kappa_std)
            output.to_csv(sub_file, index=False)
            # use cutoff
            output = self.ensemble_bagging_models_prediction(best_bagged_model_list[:(bagging_iter + 1)], best_bagged_model_weight[:(bagging_iter + 1)], cdf_test, cutoff)
            sub_file = "%s_[InitTopK%d]_[BaggingSize%d]_[BaggingFraction%s]_[Mean%.6f]_[Std%.6f]_cutoff.csv" % (subm_prefix, init_top_k, bagging_iter + 1, bagging_fraction, best_kappa_mean, best_kappa_std)
            output.to_csv(sub_file, index=False)
        return best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight
