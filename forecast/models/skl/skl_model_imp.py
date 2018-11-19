# coding=utf-8
__author__ = 'songquanwang'

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

from forecast.models.abstract_base_model import AbstractBaseModel


# 梯度自举树，也是gdbt的实现
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

import forecast.conf.model_params_conf as model_param_conf


class SklModelImp(AbstractBaseModel):
    def __init__(self, param_space, info_folder, feat_folder, feat_name):
        super(SklModelImp, self).__init__(param_space, info_folder, feat_folder, feat_name)

    def train_predict(self, param, set_obj, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        if param['task'] == "reg_skl_rf":
            pred = self.reg_skl_rf_predict(param, set_obj, all)

        elif param['task'] == "reg_skl_etr":

            pred = self.reg_skl_etr_predict(param, set_obj, all)
        elif param['task'] == "reg_skl_gbm":

            pred = self.reg_skl_gbm_predict(param, set_obj, all)
        elif param['task'] == "clf_skl_lr":

            pred = self.clf_skl_lr_predict(param, set_obj, all)
        elif param['task'] == "reg_skl_svr":
            pred = self.reg_skl_svr_predict(param, set_obj, all)

        elif param['task'] == "reg_skl_ridge":
            pred = self.reg_skl_ridge_predict(param, set_obj, all)

        elif param['task'] == "reg_skl_lasso":
            pred = self.reg_skl_lasso_predict(param, set_obj, all)
        return pred

    def reg_skl_rf_predict(self, param, set_obj, all=False):
        # regression with sklearn random forest regressor
        rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                                   max_features=param['max_features'],
                                   n_jobs=param['n_jobs'],
                                   random_state=param['random_state'])
        rf.fit(set_obj['X_train'][set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1,
               sample_weight=set_obj['weight_train'][set_obj['index_base']])
        if all:
            pred = rf.predict(set_obj['X_test'])
        else:
            pred = rf.predict(set_obj['X_valid'])

        return pred

    def reg_skl_etr_predict(self, param, set_obj, all=False):
        # regression with sklearn extra trees regressor
        etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                                  max_features=param['max_features'],
                                  n_jobs=param['n_jobs'],
                                  random_state=param['random_state'])
        etr.fit(set_obj['X_train'][set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1,
                sample_weight=set_obj['weight_train'][set_obj['index_base']])
        if all:
            pred = etr.predict(set_obj['X_test'])
        else:
            pred = etr.predict(set_obj['X_valid'])
        return pred

    def reg_skl_gbm_predict(self, param, set_obj, all=False):
        # regression with sklearn gradient boosting regressor
        gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                        max_features=param['max_features'],
                                        learning_rate=param['learning_rate'],
                                        max_depth=param['max_depth'],
                                        subsample=param['subsample'],
                                        random_state=param['random_state'])
        gbm.fit(set_obj['X_train'].toarray()[set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1,
                sample_weight=set_obj['weight_train'][set_obj['index_base']])
        if all:
            pred = gbm.predict(set_obj['X_test'].toarray())
        else:
            pred = gbm.predict(set_obj['X_valid'].toarray())
        return pred

    def clf_skl_lr_predict(self, param, set_obj, all=False):
        # classification with sklearn logistic regression   只寻找一个参数的最优参数
        lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                class_weight='balanced', random_state=param['random_state'])
        lr.fit(set_obj['X_train'][set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1)
        if all:
            pred = lr.predict_proba(set_obj['X_test'])
            w = np.asarray(range(1, model_param_conf.num_of_class + 1))
            pred = pred * w[np.newaxis, :]
            pred = np.sum(pred, axis=1)
        else:
            pred = lr.predict_proba(set_obj['X_valid'])
            w = np.asarray(range(1, model_param_conf.num_of_class + 1))
            pred = pred * w[np.newaxis, :]
            pred = np.sum(pred, axis=1)
        return pred

    def reg_skl_svr_predict(self, param, set_obj, all=False):
        # regression with sklearn support vector regression
        X_train = set_obj['X_train'].toarray()
        scaler = StandardScaler()
        X_train[set_obj['index_base']] = scaler.fit_transform(X_train[set_obj['index_base']])
        svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'],
                  degree=param['degree'], kernel=param['kernel'])
        svr.fit(X_train[set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1,
                sample_weight=set_obj['weight_train'][set_obj['index_base']])
        if all:
            X_test = set_obj['X_test'].toarray()
            X_test = scaler.transform(X_test)
            pred = svr.predict(X_test)
        else:
            X_valid = set_obj['X_valid'].toarray()
            X_valid = scaler.transform(X_valid)
            pred = svr.predict(X_valid)
        return pred

    def reg_skl_ridge_predict(self, param, set_obj, all=False):
        # regression with sklearn ridge regression
        ridge = Ridge(alpha=param["alpha"], normalize=True)
        ridge.fit(set_obj['X_train'][set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1,
                  sample_weight=set_obj['weight_train'][set_obj['index_base']])
        if all:
            pred = ridge.predict(set_obj['X_test'])
        else:
            pred = ridge.predict(set_obj['X_valid'])
        return pred

    def reg_skl_lasso_predict(self, param, set_obj, all=False):
        # regression with sklearn lasso
        lasso = Lasso(alpha=param["alpha"], normalize=True)
        lasso.fit(set_obj['X_train'][set_obj['index_base']], set_obj['labels_train'][set_obj['index_base']] + 1)
        if all:
            pred = lasso.predict(set_obj['X_test'])
        else:
            pred = lasso.predict(set_obj['X_valid'])
        return pred

    @staticmethod
    def get_id():
        return "skl_model_id"

    @staticmethod
    def get_name():
        return "skl_model"
