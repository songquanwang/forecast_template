# coding=utf-8
__author__ = 'songquanwang'

import os

import numpy as np

from forecast.models.abstract_base_model import AbstractBaseModel

## sklearn
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import StandardScaler

import forecast.conf.model_params_conf as model_param_conf


class LibfmModelImp(AbstractBaseModel):
    def __init__(self, param_space, info_folder, feat_folder, feat_name):
        super(LibfmModelImp, self).__init__(param_space, info_folder, feat_folder, feat_name)

    def train_predict(self, param, set_obj, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        # scale
        scaler = StandardScaler()
        X_train = set_obj['X_train'].toarray()
        X_train[set_obj['index_base']] = scaler.fit_transform(X_train[set_obj['index_base']])
        # dump feat
        train_tmp_path = "%s.tmp" % set_obj['feat_train_path']
        dump_svmlight_file(X_train[set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']], train_tmp_path)
        if all:
            X_test = scaler.transform(set_obj['X_test'].toarray())
            labels_test = set_obj['labels_test']
            test_tmp_path = "%s.tmp" % set_obj['feat_test_path']
            raw_pred_test_path = set_obj['raw_pred_test_path']
        else:
            X_test = scaler.transform(set_obj['X_valid'].toarray())
            labels_test = set_obj['labels_valid']
            test_tmp_path = "%s.tmp" % set_obj['feat_valid_path']
            raw_pred_test_path = set_obj['raw_pred_valid_path']

        dump_svmlight_file(X_test, labels_test, test_tmp_path)
        # train fm
        cmd = "%s -task r -train %s -test %s -out %s -dim '1,1,%d' -iter %d > libfm.log" % \
              (model_param_conf.libfm_exe, train_tmp_path, test_tmp_path, raw_pred_test_path, param['dim'], param['iter'])
        os.system(cmd)
        os.remove(train_tmp_path)
        os.remove(test_tmp_path)
        pred = np.loadtxt(raw_pred_test_path, dtype=float)
        # labels are in [0,1,2,3]
        pred += 1

        return pred

    @staticmethod
    def get_id():
        return "libfm_model_id"

    @staticmethod
    def get_name():
        return "libfm_model"
