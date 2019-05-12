# -*- coding: utf-8 -*-
"""
  @Author: zzn 
  @Date: 2019-04-17 19:34:38 
  @Last Modified by:   zzn 
  @Last Modified time: 2019-04-17 19:34:38 
"""
from time import gmtime, strftime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import gen_features_sqw


def eval_f(y_pred, train_data):
    y_true = train_data.label
    # 分类在前，行号在后
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def submit_result(submit, result, model_name):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv('../submit/{}_result_{}.csv'.format(model_name, now_time), index=False)


def train_lgb(train_x, train_y, test_x, cate_cols):
    """
    训练并保存模型
    :param train_x:
    :param train_y:
    :param test_x:
    :return:
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
    }
    # cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
    #              'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']
    scores = []
    result_proba = []
    i = 0
    for tr_idx, val_idx in kfold.split(train_x, train_y):
        print('#######################################{}'.format(i))
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=cate_cols)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=cate_cols)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=500, num_boost_round=40000, verbose_eval=50,
                              feval=eval_f)
        lgb_model.save_model('../models/model_{}'.format(i))
        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        result_proba.append(lgb_model.predict(
            test_x, num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
        i += 1
    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    return pred_test


def tp_submit():
    """
    训练模型，生成测试集预测结果
    :return:
    """
    train_x, train_y, test_x, test_y, train_sid, test_sid, cate_cols = gen_features_sqw.get_train_test_feas_submit()
    result_lgb = train_lgb(train_x, train_y, test_x, cate_cols)
    submit_result(test_sid, result_lgb, 'lgb')


def tp_valid():
    """
    训练模型，生成验证集预测结果
    :return:
    """
    train_x, train_y, test_x, test_y, train_sid, test_sid, cate_cols = gen_features_sqw.get_train_test_feas_valid()
    result_lgb = train_lgb(train_x, train_y, test_x, cate_cols)
    submit_result(test_sid, result_lgb, 'lgb')


def predict_by_model():
    """
    预测训练结果
    :return:
    """
    train_x, train_y, test_x, test_y, train_sid, test_sid, cate_cols = gen_features_sqw.get_train_test_feas_valid()
    result_proba = []
    scores = []
    for i in range(5):
        print('***************************{}'.format(i))
        lgb_model = lgb.Booster(model_file='../models/model_{}'.format(i))
        pred_onehot = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
        test_pred = np.argmax(pred_onehot, axis=1)
        val_score = f1_score(test_y, test_pred, average='weighted')
        result_proba.append(pred_onehot)
        scores.append(val_score)

    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    result_df = pd.DataFrame()
    result_df['sid'] = train_sid.sid
    result_df['recommend_mode'] = pred_test

    result_df.to_csv('../submit/lgb_train_result.csv', index=False)


if __name__ == '__main__':
    tp_submit()
