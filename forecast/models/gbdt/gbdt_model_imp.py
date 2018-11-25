# coding=utf-8
__author__ = 'songquanwang'

import numpy as np

import xgboost as xgb
from forecast.models.abstract_base_model import AbstractBaseModel
import forecast.conf.model_params_conf as model_param_conf
import forecast.utils.utils as utils


class GbdtModelImp(AbstractBaseModel):
    def __init__(self, param_space, info_folder, feat_folder, feat_name):
        super(GbdtModelImp, self).__init__(param_space, info_folder, feat_folder, feat_name)

    def train_predict(self, param, set_obj, all=False):
        """
        数据训练
        :param train_end_date:
        :return:
        """
        if param["task"] in ["regression", "ranking"]:
            pred = self.reg_rank_predict(param, set_obj, all)
        elif param["task"] in ["softmax"]:
            pred = self.soft_max_predict(param, set_obj, all)
        elif param["task"] in ["softkappa"]:
            pred = self.soft_softkappa_predict(param, set_obj, all)
        elif param["task"] in ["ebc"]:
            pred = self.ebc_predict(param, set_obj, all)
        elif param["task"] in ["cocr"]:
            pred = self.cocr_predict(param, set_obj, all)

        return pred

    def reg_rank_predict(self, param, set_obj, all=False):
        # regression & pairwise ranking with xgboost
        if all:
            evalerror_regrank_test = lambda preds, dtrain: utils.evalerror_regrank_cdf(preds, dtrain, set_obj['cdf_test'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], feval=evalerror_regrank_test)
            pred = bst.predict(set_obj['dtest'])

        else:
            evalerror_regrank_valid = lambda preds, dtrain: utils.evalerror_regrank_cdf(preds, dtrain, set_obj['cdf_valid'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], feval=evalerror_regrank_valid)
            pred = bst.predict(set_obj['dvalid'])

        return pred

    def soft_max_predict(self, param, set_obj, all=False):
        # softmax regression with xgboost
        if all:
            evalerror_softmax_test = lambda preds, dtrain: utils.evalerror_softmax_cdf(preds, dtrain, set_obj['cdf_test'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], feval=evalerror_softmax_test)
            pred = bst.predict(set_obj['dtest'])
            w = np.asarray(range(1, model_param_conf.num_of_class + 1))
        else:
            evalerror_softmax_valid = lambda preds, dtrain: utils.evalerror_softmax_cdf(preds, dtrain, set_obj['cdf_valid'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], feval=evalerror_softmax_valid)
            # (6688, 4)
            pred = bst.predict(set_obj['dvalid'])

        # softprob结果是ndata * nclass矩阵，每个列值为样本所属于每个类别的概率
        pred = pred * w[np.newaxis, :]
        pred = np.sum(pred, axis=1)
        return pred

    def soft_softkappa_predict(self, param, set_obj, all=False):
        # softkappa with xgboost
        if all:
            evalerror_softkappa_test = lambda preds, dtrain: utils.evalerror_softkappa_cdf(preds, dtrain, set_obj['cdf_test'])
            obj = lambda preds, dtrain: utils.softkappaObj(preds, set_obj['dtrain'], hess_scale=param['hess_scale'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], obj=obj, feval=evalerror_softkappa_test)
            pred = utils.softmax(bst.predict(set_obj['dtest']))

        else:
            evalerror_softkappa_valid = lambda preds, dtrain: utils.evalerror_softkappa_cdf(preds, dtrain, set_obj['cdf_valid'])
            obj = lambda preds, dtrain: utils.softkappaObj(preds, dtrain, hess_scale=param['hess_scale'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], obj=obj, feval=evalerror_softkappa_valid)
            pred = utils.softmax(bst.predict(set_obj['dvalid']))

        w = np.asarray(range(1, model_param_conf.num_of_class + 1))
        pred = pred * w[np.newaxis, :]
        pred = np.sum(pred, axis=1)
        return pred

    def ebc_predict(self, param, set_obj, all=False):
        # ebc with xgboost
        if all:
            evalerror_ebc_test = lambda preds, dtrain: utils.evalerror_ebc_cdf(preds, dtrain, set_obj['cdf_test'], model_param_conf.ebc_hard_threshold)
            obj = lambda preds, dtrain: utils.ebcObj(preds, dtrain)
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], obj=obj, feval=evalerror_ebc_test)
            pred = utils.sigmoid(bst.predict(set_obj['dtest']))

        else:
            evalerror_ebc_valid = lambda preds, dtrain: utils.evalerror_ebc_cdf(preds, dtrain, set_obj['cdf_valid'], model_param_conf.ebc_hard_threshold)
            obj = lambda preds, dtrain: utils.ebcObj(preds, dtrain)
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], obj=obj, feval=evalerror_ebc_valid)
            pred = utils.sigmoid(bst.predict(set_obj['dvalid']))
        pred = utils.applyEBCRule(pred, hard_threshold=utils.ebc_hard_threshold)
        return pred

    def cocr_predict(self, param, set_obj, all=False):
        # cocr with xgboost
        if all:
            evalerror_cocr_test = lambda preds, dtrain: utils.evalerror_cocr_cdf(preds, dtrain, set_obj['cdf_test'])
            obj = lambda preds, dtrain: utils.cocrObj(preds, dtrain)
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], obj=obj, feval=evalerror_cocr_test)
            pred = bst.predict(set_obj['dtest'])
        else:
            evalerror_cocr_valid = lambda preds, dtrain: utils.evalerror_cocr_cdf(preds, dtrain, set_obj['cdf_valid'])
            obj = lambda preds, dtrain: utils.cocrObj(preds, set_obj['dtrain'])
            bst = xgb.train(param, set_obj['dtrain'], param['num_round'], set_obj['watchlist'], obj=obj, feval=evalerror_cocr_valid)
            pred = bst.predict(set_obj['dvalid'])

        pred = utils.applyCOCRRule(pred)
        return pred

    @staticmethod
    def get_id():
        return "gdbt_model_id"

    @staticmethod
    def get_name():
        return "gdbt_model"


if __name__ == "__main__":
    pass
