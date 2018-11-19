# coding=utf-8
__author__ = 'songquanwang'

import os

import numpy as np

from forecast.models.abstract_base_model import AbstractBaseModel
import forecast.conf.model_params_conf as model_param_conf


class RgfModelImp(AbstractBaseModel):
    def __init__(self, param_space, info_folder, feat_folder, feat_name):
        super(RgfModelImp, self).__init__(param_space, info_folder, feat_folder, feat_name)

    def train_predict(self, param, set_obj, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        # to array
        X_train = set_obj['X_train'].toarray()
        train_x_fn = set_obj['feat_train_path'] + ".x"
        train_y_fn = set_obj['feat_train_path'] + ".y"
        model_fn_prefix = "rgf_model"
        np.savetxt(train_x_fn, X_train[set_obj['index_base']], fmt="%.6f", delimiter='\t')
        np.savetxt(train_y_fn, set_obj['labels_train'][set_obj['index_base']], fmt="%d", delimiter='\t')

        if all:
            X_test = set_obj['X_test'].toarray()
            test_x_fn = set_obj['feat_test_path'] + ".x"
            test_pred_fn = set_obj['feat_test_path'] + ".pred"
            # np.savetxt(test_y_fn, set_obj['labels_test, fmt="%d", delimiter='\t')
        else:
            X_test = set_obj['X_valid'].toarray()
            test_x_fn = set_obj['feat_valid_path'] + ".x"
            test_pred_fn = set_obj['feat_valid_path'] + ".pred"
            # np.savetxt(valid_y_fn, set_obj['labels_valid, fmt="%d", delimiter='\t')
        np.savetxt(test_x_fn, X_test, fmt="%.6f", delimiter='\t')
        pars = [
            "train_x_fn=", train_x_fn, "\n",
            "train_y_fn=", train_y_fn, "\n",
            # "train_w_fn=",weight_train_path,"\n",
            "model_fn_prefix=", model_fn_prefix, "\n",
            "reg_L2=", param['reg_L2'], "\n",
            # "reg_depth=", 1.01, "\n",
            "algorithm=", "RGF", "\n",
            "loss=", "LS", "\n",
            # "opt_interval=", 100, "\n",
            "test_interval=", param['max_leaf_forest'], "\n",
            "max_leaf_forest=", param['max_leaf_forest'], "\n",
            "num_iteration_opt=", param['num_iteration_opt'], "\n",
            "num_tree_search=", param['num_tree_search'], "\n",
            "min_pop=", param['min_pop'], "\n",
            "opt_interval=", param['opt_interval'], "\n",
            "opt_stepsize=", param['opt_stepsize'], "\n",
            "NormalizeTarget"
        ]
        pars = "".join([str(p) for p in pars])
        rfg_setting_train = "./rfg_setting_train"
        with open(rfg_setting_train + ".inp", "wb") as f:
            f.write(pars)
        # train fm
        cmd = "perl %s %s train %s >> rgf.log" % (
            model_param_conf.call_exe, model_param_conf.rgf_exe, rfg_setting_train)
        # print cmd
        os.system(cmd)
        model_fn = model_fn_prefix + "-01"
        pars = [
            "test_x_fn=", test_x_fn, "\n",
            "model_fn=", model_fn, "\n",
            "prediction_fn=", test_pred_fn
        ]
        pars = "".join([str(p) for p in pars])
        rfg_setting_test = "./rfg_setting_test"
        with open(rfg_setting_test + ".inp", "wb") as f:
            f.write(pars)
        cmd = "perl %s %s predict %s >> rgf.log" % (
            model_param_conf.call_exe, model_param_conf.rgf_exe, rfg_setting_test)
        # print cmd
        os.system(cmd)
        pred = np.loadtxt(test_pred_fn, dtype=float)

        return pred

    @staticmethod
    def get_id():
        return "rgf_model_id"

    @staticmethod
    def get_name():
        return "rgf_model"
