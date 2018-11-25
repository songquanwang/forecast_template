# coding=utf-8
__author__ = 'songquanwang'
import os

import numpy as np

from hyperopt import fmin, tpe, Trials
from forecast.models.gbdt.gbdt_model_imp import GbdtModelImp
from forecast.models.keras.keras_dnn_model_imp import KerasDnnModelImp
from forecast.models.libfm.libfm_model_imp import LibfmModelImp
from forecast.models.rgf.rgf_model_imp import RgfModelImp
from forecast.models.skl.skl_model_imp import SklModelImp
import forecast.conf.model_params_conf as config
import forecast.conf.model_library_config  as model_library_config


def create_model(param_space, info_folder, feat_folder, feat_name):
    """
    模型工厂，根据参数创建模型对象
    :param param_space:
    :param feat_folder:
    :param feat_name:
    :return: param_space, info_folder, feat_folder, feat_name
    """
    if param_space["task"] in ["regression", "ranking", "softmax", "softkappa", "ebc", "cocr"]:
        return GbdtModelImp(param_space, info_folder, feat_folder, feat_name)
    elif param_space["task"] in ["reg_skl_rf", "reg_skl_etr", "reg_skl_gbm", "clf_skl_lr", "reg_skl_svr", "reg_skl_ridge", "reg_skl_lasso"]:
        return SklModelImp(param_space, info_folder, feat_folder, feat_name)
    elif param_space["task"] in ["reg_keras_dnn"]:
        return KerasDnnModelImp(param_space, info_folder, feat_folder, feat_name)
    elif param_space["task"] in ["reg_libfm"]:
        return LibfmModelImp(param_space, info_folder, feat_folder, feat_name)
    elif param_space["task"] in ["reg_rgf"]:
        return RgfModelImp(param_space, info_folder, feat_folder, feat_name)
    else:
        raise Exception('暂时不支持该模型!')


def make_opt_predict_by_models(specified_models):
    """
    使用指定的模型预测结果
    所有尝试的参数均记录在文件中
    :param specified_models:
    :return:best_kappa_mean, best_kappa_std
    """
    log_path = "%s/Log" % config.output_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    models_best_params = []
    # 判断传入参数中的models是不是已经配置的models
    for feat_name in specified_models:
        if feat_name not in model_library_config.feat_names:
            continue
        # param space ,模型内部也需要（打日志头部)
        feat_folder, param_space = model_library_config.model_config[feat_name]
        model = create_model(param_space, config.solution_info, feat_folder, feat_name)
        model.log_header()

        print("************************************************************")
        print("Search for the best params")
        # global trial_counter
        trials = Trials()
        objective = lambda p: model.hyperopt_obj(p, feat_folder, feat_name)
        best_params = fmin(objective, param_space, algo=tpe.suggest, trials=trials, max_evals=param_space["max_evals"])
        # 把best_params包含的数字属性转成int
        for f in model_library_config.int_feat:
            if best_params.has_key(f):
                best_params[f] = int(best_params[f])
        print("************************************************************")
        print("Best params")
        for k, v in best_params.items():
            print("        %s: %s" % (k, v))
        # 获取尝试的losses
        trial_kappas = -np.asarray(trials.losses(), dtype=float)
        best_kappa_mean = max(trial_kappas)
        # where返回两个维度的坐标
        ind = np.where(trial_kappas == best_kappa_mean)[0][0]
        # 找到最优参数的std
        best_kappa_std = trials.trial_attachments(trials.trials[ind])['std']
        print("Kappa stats")
        print("Mean: %.6f\n        Std: %.6f" % (best_kappa_mean, best_kappa_std))

        models_best_params.append((feat_name, best_kappa_mean, best_kappa_std))

    return models_best_params


if __name__ == "__main__":
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
    models_best_params = make_opt_predict_by_models(specified_models)
    for model_name, best_kappa_mean, best_kappa_std in models_best_params:
        print("Model:%s Mean: %.6f\n Std: %.6f" % (model_name, best_kappa_mean, best_kappa_std))
