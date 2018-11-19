# coding:utf-8
"""
整体流程
1.预处理
2.生成特征；合并特征
3.生成统计信息文件
4.模型训练、预测结果保存
5.集成训练结果
"""
__author__ = 'songquanwang'

import numpy as np
from forecast.preprocess.preprocess import preprocess
from forecast.preprocess.init_path import init_path
from forecast.preprocess.kfold import gen_stratified_kfold
from forecast.info.gen_info import gen_info

import forecast.conf.feat.LSA_and_stats_feat_Jun09_Low as LSA_and_stats_feat_Jun09_Low
import forecast.conf.feat.LSA_svd150_and_Jaccard_coef_Jun14_Low as LSA_svd150_and_Jaccard_coef_Jun14_Low
import forecast.conf.feat.svd100_and_bow_Jun23_Low as svd100_and_bow_Jun23_Low
import forecast.conf.feat.svd100_and_bow_Jun27_High as svd100_and_bow_Jun27_High

from forecast.feat.abstract_base_feat import AbstractBaseFeat
from forecast.feat.basic_tfidf_feat import BasicTfidfFeat
from forecast.feat.cooccurrence_tfidf_feat import CooccurenceTfidfFeat
from forecast.feat.counting_feat import CountingFeat
from forecast.feat.distance_feat import DistanceFeat
from forecast.feat.id_feat import IdFeat

import forecast.conf.model_library_config as model_library_config
import forecast.conf.model_params_conf as config
import forecast.models.model_manager as model_manager

from forecast.ensemble.predict_ensemble import PredictEnsemble


def init():
    """
    预处理
     1.构建必要的目录
     2.预处理
     3.交叉验证
    :return:
    """
    init_path()
    # preprocess()
    # gen_stratified_kfold()


def gen_info():
    """
    创建info文件
    :return:
    """
    # 生成数据的统计信息，这些info跟特征无关
    gen_info()


def gen_feat():
    # 不仅生成特征文件，还生成四个特征文件名字的文件 在Feat/solution/counting.feat_name等..
    # 生成所有的特征+label
    # 生成basic tfidf feat
    # basic_tfidf_feat = BasicTfidfFeat()
    # basic_tfidf_feat.gen_feat_cv()
    # # 生成coocrrence tfidf feat
    # cooccurence_tfidf_feat = CooccurenceTfidfFeat()
    # cooccurence_tfidf_feat.gen_feat_cv()
    # # 生成 counting feat
    # counting_feat = CountingFeat()
    # counting_feat.gen_feat_cv()
    # # 生成 distance feat
    # distance_feat = DistanceFeat()
    # distance_feat.gen_feat_cv()
    # id_feat = IdFeat()
    # id_feat.gen_feat_cv()

    # 合并所有的feat 生成四个目录，文件名字 train.feat valid.feat test.feat
    AbstractBaseFeat.extract_feats_cv(LSA_and_stats_feat_Jun09_Low.feat_names, feat_path_name="LSA_and_stats_feat_Jun09")

    AbstractBaseFeat.extract_feats_cv(LSA_svd150_and_Jaccard_coef_Jun14_Low.feat_names, feat_path_name="LSA_svd150_and_Jaccard_coef_Jun14")

    AbstractBaseFeat.extract_feats_cv(svd100_and_bow_Jun23_Low.feat_names, feat_path_name="svd100_and_bow_Jun23")

    AbstractBaseFeat.extract_feats_cv(svd100_and_bow_Jun27_High.feat_names, feat_path_name="svd100_and_bow_Jun27")


def predict():
    """
    使用指定的模型预测结果
    :param specified_models:
    :return:best_kappa_mean, best_kappa_std
    """
    specified_models = [
        # LSA_and_stats_feat_Jun09 (Low)
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_xgb_tree]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_xgb_linear]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@cocr_xgb_linear]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@kappa_xgb_linear]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_skl_etr]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_skl_rf]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_skl_gbm]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_skl_svr]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_skl_ridge]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_skl_lasso]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@clf_skl_lr]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_libfm]",
        "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_keras_dnn]",
        # "[Pre@solution]_[Feat@LSA_and_stats_feat_Jun09]_[Model@reg_rgf]"
    ]
    models_best_params = model_manager.make_opt_predict_by_models(specified_models)
    for model_name, best_kappa_mean, best_kappa_std in models_best_params:
        print("Model:%s Mean: %.6f\n Std: %.6f" % (model_name, best_kappa_mean, best_kappa_std))


def ensemble():
    """
    "../../Feat/solution/LSA_and_stats_feat_Jun09"
    :return:
    """
    feat_folder = model_library_config.feat_folders[0]
    model_folder = feat_folder
    subm_folder = "./solution1/Output/ensenbled_subm"
    predict_ensemble = PredictEnsemble(model_folder, subm_folder)
    cdf_test = np.loadtxt("%s/All/test.cdf" % config.solution_info, dtype=float)
    cdf_valid = None
    bagging_size = 100
    # 选择全部模型
    bagging_fraction = 1.0
    # 剪枝参数没有用到
    prunning_fraction = 1.
    bagging_replacement = True
    init_top_k = 5
    hypteropt_max_evals = 1
    w_min = -1
    w_max = 1
    subm_prefix = "%s/test.pred.[ensemble_selection]_[Solution]" % (subm_folder)
    best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight = predict_ensemble.ensemble_model_list_pedicts(feat_folder, cdf_valid, cdf_test,
                                                                                                                                     subm_prefix,
                                                                                                                                     hypteropt_max_evals, w_min, w_max,
                                                                                                                                     bagging_replacement=bagging_replacement,
                                                                                                                                     bagging_fraction=bagging_fraction,
                                                                                                                                     bagging_size=bagging_size,
                                                                                                                                     init_top_k=init_top_k,
                                                                                                                                     prunning_fraction=prunning_fraction)
    print("best_kappa_mean: %.6f\n best_kappa_std: %.6f\n  best_bagged_model_list: %r \n best_bagged_model_weight: %r \n " % (
        best_kappa_mean, best_kappa_std, best_bagged_model_list, best_bagged_model_weight))


if __name__ == "__main__":
    # gen_feat()
    # predict()
    init()
