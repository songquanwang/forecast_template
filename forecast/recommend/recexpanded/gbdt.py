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
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import gen_features_sqw_expanded


def eval_f(y_pred, train_data):
    y_true = train_data.label
    # 分类在前，行号在后
    # y_pred = y_pred.reshape((12, -1)).T
    # y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def score_logloss(y_true, y_pred):
    """
    平均对数损失
    :param y_true:
    :param y_pred:
    :return:
    """
    score = log_loss(y_true, y_pred)
    return score


def score_accurate_rate(y_pred, train_sid_valid):
    """
    预测准确率
    :param y_true:
    :param y_pred:
    :param train_sid_valid:  'sid', 'click_mode', 'transport_mode'
    :return:
    """
    train_sid_valid['mode_pro'] = y_pred
    train_sid_valid['max_pro'] = train_sid_valid.groupby(['sid'])['mode_pro'].transform(np.max)
    final_submit = train_sid_valid.loc[train_sid_valid['mode_pro'] == train_sid_valid['max_pro'], ['sid', 'click_mode', 'transport_mode']].drop_duplicates(subset=['sid'])
    correct_num = len(final_submit[final_submit['click_mode'] == final_submit['transport_mode']])
    all_num = len(final_submit)
    score = correct_num / all_num
    print('correct_num is:{}  all_num is :{}'.format(correct_num, all_num))
    return score


def submit_result(submit, result, file_name):
    """
    转换格式，生成提交文件
    :param submit:
    :param result:
    :param model_name:
    :param file_name:
    :return:
    """
    submit['mode_pro'] = result
    submit['max_pro'] = submit.groupby(['sid'])['mode_pro'].transform(np.max)
    final_submit = submit.loc[submit['mode_pro'] == submit['max_pro'], ['sid', 'transport_mode']].drop_duplicates(subset=['sid'])
    final_submit = final_submit.rename(columns={'transport_mode': 'recommend_mode'})
    final_submit.to_csv(file_name, index=False)
    return final_submit


def train_lgb(train_x, train_y, test_x, train_sid, test_sid, cate_cols):
    """
    训练并保存模型
    :param train_x:
    :param train_y:
    :param test_x:
    :return:
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'xentropy',
        'metrics': 'xentropy',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        # 'num_class': 12,
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
        tr_x, tr_y, val_x, val_y, train_sid_valid = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx], train_sid.iloc[val_idx]
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=cate_cols)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=cate_cols)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=500, num_boost_round=40000, verbose_eval=50)
        lgb_model.save_model('../models/model_{}'.format(i))
        val_pred = lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration)
        val_score = score_accurate_rate(val_pred, train_sid_valid)
        result_proba.append(lgb_model.predict(
            test_x, num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
        i += 1
    print('cv accurate rate score: ', np.mean(scores))
    pred_test = np.mean(result_proba, axis=0)
    return pred_test


def tp_submit():
    """
    训练模型，生成测试集预测结果
    :return:
    """
    train_x, train_y, test_x, test_y, train_sid, test_sid, cate_cols = gen_features_sqw_expanded.get_train_test_feas_submit()
    result_lgb = train_lgb(train_x, train_y, test_x, train_sid, test_sid, cate_cols)
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}.csv'.format('lgb', now_time)
    submit_result(test_sid, result_lgb, file_name)


def tp_valid():
    """
    训练模型，生成验证集预测结果
    :return:
    """
    train_x, train_y, test_x, test_y, train_sid, test_sid, cate_cols = gen_features_sqw_expanded.get_train_test_feas_valid()
    result_lgb = train_lgb(train_x, train_y, test_x, train_sid, test_sid, cate_cols)
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}_valid.csv'.format('lgb', now_time)
    test_sid['click_mode'] = test_sid['click_mode'].astype(np.int32)
    test_sid['mode_pro'] = result_lgb
    test_sid['max_pro'] = test_sid.groupby(['sid'])['mode_pro'].transform(np.max)
    final_submit = test_sid.loc[test_sid['mode_pro'] == test_sid['max_pro'], ['sid', 'transport_mode', 'click_mode']].drop_duplicates(subset=['sid'])
    final_submit = final_submit.rename(columns={'transport_mode': 'recommend_mode'})
    final_submit.to_csv(file_name, index=False)


def predict_by_model():
    """
    预测训练结果
    :return:
    """
    train_x, train_y, test_x, test_y, train_sid, test_sid, cate_cols = gen_features_sqw_expanded.get_train_test_feas_valid()
    result_proba = []
    scores = []
    for i in range(5):
        print('***************************{}'.format(i))
        lgb_model = lgb.Booster(model_file='../models/model_{}'.format(i))
        test_pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
        val_score = score_accurate_rate(test_pred, test_sid)
        result_proba.append(test_pred)
        scores.append(val_score)

    print('cv accurate_rate-score: ', np.mean(scores))
    pred_test = np.mean(result_proba, axis=0)

    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}.csv'.format('lgb', now_time)
    submit_result(test_sid, pred_test, file_name)


if __name__ == '__main__':
    tp_valid()
