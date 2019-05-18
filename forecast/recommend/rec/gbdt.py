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
from sklearn.model_selection import train_test_split
import conf


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


def get_train_test_feats():
    """
    获取训练数据、测试数据
    :return:
    """
    data = pd.read_csv('../data/features/features_all_ts.csv')
    train_df = data[data['click_mode'] != -1]
    test_df = data[data['click_mode'] == -1]
    return train_df, test_df


def get_train_valid_feats():
    """
    获取训练数据、验证数据
    :return:
    """

    data = pd.read_csv('../data/features/features_all_ts.csv')
    train_df = data[data['click_mode'] != -1]
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    # train_df_t, train_df_e = next(kfold.split(train_df, train_df['click_mode']))
    # train_df.iloc[train_df_t], train_df.iloc[train_df_e]
    train_df_t, train_df_e = train_test_split(train_df, test_size=0.2)

    return train_df_t, train_df_e


def tp_submit():
    """
    训练模型，生成测试集预测结果
    :return:
    """
    train_df, test_df = get_train_test_feats()
    result_lgb = train_lgb(train_df, test_df)
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}.csv'.format('gbdt_ext', now_time)
    submit = test_df[['sid']]
    submit['recommend_mode'] = result_lgb
    if len(submit) < 94358:
        empty_pred_df = pd.read_csv('../submit/empty_pred.csv')
        submit = pd.concat([submit, empty_pred_df], axis=0)
    submit.to_csv(file_name, index=False)


def tp_valid():
    """
    训练模型，生成验证集预测结果
    :return:
    """
    train_df, valid_df = get_train_valid_feats()
    result_lgb = train_lgb(train_df, valid_df)
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}.csv'.format('gbdt_ext_valid', now_time)
    submit = valid_df[['sid', 'click_mode']]
    submit['recommend_mode'] = result_lgb
    submit.to_csv(file_name, index=False)


def train_lgb(train_df, test_df):
    """
    训练并保存模型
    :param train_df:
    :param test_df:
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
        'verbose': -1
    }
    # cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
    #              'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour']
    scores = []
    result_proba = []
    i = 0
    for tr_idx, val_idx in kfold.split(train_df, train_df['click_mode']):
        print('#######################################{}'.format(i))
        train_data = train_df.iloc[tr_idx]
        valid_data = train_df.iloc[val_idx]
        tr_x, tr_y, val_x, val_y = train_data[conf.feature_columns], train_data['click_mode'], \
                                   valid_data[conf.feature_columns], valid_data['click_mode']
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=conf.cate_columns)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=conf.cate_columns)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=500, num_boost_round=40000, verbose_eval=50,
                              feval=eval_f)
        lgb_model.save_model('../models/model_{}'.format(i))
        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        result_proba.append(lgb_model.predict(test_df[conf.feature_columns], num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
        i += 1
    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    return pred_test


def predict_by_model():
    """
    预测训练结果
    :return:
    """
    train_df, valid_df = get_train_test_feats()
    # train_df, valid_df = get_train_valid_feats()
    result_proba = []
    scores = []
    for i in range(1):
        print('***************************{}'.format(i))
        lgb_model = lgb.Booster(model_file='../models/model_{}'.format(i))
        pred_onehot = lgb_model.predict(valid_df[conf.feature_columns], num_iteration=lgb_model.best_iteration)
        test_pred = np.argmax(pred_onehot, axis=1)
        val_score = f1_score(valid_df['click_mode'], test_pred, average='weighted')
        result_proba.append(pred_onehot)
        scores.append(val_score)

    print('cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    result_df = pd.DataFrame()
    result_df['sid'] = valid_df.sid
    # result_df['click_mode'] = valid_df.click_mode
    result_df['recommend_mode'] = pred_test
    if len(result_df) < 94358:
        empty_pred_df = pd.read_csv('../submit/empty_pred.csv')
        result_df = pd.concat([result_df, empty_pred_df], axis=0)
    result_df.to_csv('../submit/2019-05-17-pre_feat.csv', index=False)


if __name__ == '__main__':
    tp_submit()
    #predict_by_model()
