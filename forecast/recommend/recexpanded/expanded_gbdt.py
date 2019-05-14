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

import expanded_conf as conf

zero_threshold = 0.5


def split_data_by_clickmode(df, kfold):
    """
    按照某个字段分层抽样
    :param df:
    :param columns:
    :return:
    """

    # 先去重，保证sid唯一
    group_df = df[['sid', 'click_mode']].drop_duplicates()
    train_click_mode_idx, valid_click_mode_idx = next(kfold.split(group_df, group_df['click_mode']))
    # 保证训练集、验证集 sid没有交集
    train_sids, valid_sids = group_df.iloc[train_click_mode_idx].sid, group_df.iloc[valid_click_mode_idx].sid
    train_data = df[df['sid'].isin(train_sids)]
    valid_data = df[df['sid'].isin(valid_sids)]
    return train_data, valid_data


def convert_plans_pro_to_pred(test_df, y_pred, threshold=0.5):
    """
    把各个plan的 binary 概率预测转成 推荐模式预测
    :param test_df:
    :param y_pred:
    :param threshold:
    :return:
    """
    test_df.loc[:, 'mode_pro'] = y_pred
    test_df.loc[:, 'max_pro'] = test_df.groupby(['sid'])['mode_pro'].transform(np.max)
    final_submit_df = test_df.loc[test_df['mode_pro'] == test_df['max_pro'], ['sid', 'click_mode', 'transport_mode', 'max_pro']].drop_duplicates(subset=['sid'])
    # 按照阈值，把低概率预测为0
    final_submit_df.loc[final_submit_df['max_pro'] < threshold, 'transport_mode'] = 0
    return final_submit_df[['sid', 'click_mode', 'transport_mode']]


def eval_f1(train_data, y_pred):
    """
    计算f1_score
    :param train_data:
    :param y_pred:
    :return:
    """
    score = score_f1(train_data, y_pred)
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


def score_f1(train_data, y_pred):
    """
    计算f1 score
    :return:
    """
    pred_df = convert_plans_pro_to_pred(train_data, y_pred, threshold=zero_threshold)
    score = f1_score(pred_df['click_mode'], pred_df['transport_mode'], average='weighted')
    return score


def score_accurate_rate(test_df, y_pred):
    """
    预测准确率
    :param y_true:
    :param y_pred:
    :param train_sid_valid:  'sid', 'click_mode', 'transport_mode'
    :return:
    """
    final_submit = convert_plans_pro_to_pred(test_df, y_pred, threshold=zero_threshold)
    correct_num = len(final_submit[final_submit['click_mode'] == final_submit['transport_mode']])
    all_num = len(final_submit)
    score = correct_num / all_num
    print('correct_num is:{}  all_num is :{}'.format(correct_num, all_num))
    return score


def submit_result(test_df, result, file_name):
    """
    转换格式，生成提交文件
    :param submit:
    :param result:
    :param model_name:
    :param file_name:
    :return:
    """
    cut_df = test_df[['sid', 'click_mode', 'transport_mode']]
    submit = convert_plans_pro_to_pred(cut_df, result, threshold=0.5)
    final_submit = submit.rename(columns={'transport_mode': 'recommend_mode'})
    final_submit[['sid', 'recommend_mode']].to_csv(file_name, index=False)
    return final_submit


def train_lgb(train_df, test_df):
    """
    训练并保存模型
    :param train_x:
    :param train_y:
    :param test_x:
    :return:
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'binary',
        'metrics': 'binary',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        # 'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'verbose': -1
    }
    group_df = train_df[['sid', 'click_mode']].drop_duplicates()
    acc_scores = []
    f1_scores = []
    result_proba = []
    i = 0
    for tr_idx, val_idx in kfold.split(group_df, group_df['click_mode']):
        print('#######################################{}'.format(i))
        train_sids, valid_sids = group_df.iloc[tr_idx].sid, group_df.iloc[val_idx].sid
        train_data = train_df.loc[train_df['sid'].isin(train_sids)]
        valid_data = train_df.loc[train_df['sid'].isin(valid_sids)]
        tr_x, tr_y, val_x, val_y, train_sid_train, train_sid_valid = train_data[conf.feature_columns], train_data['is_click'], \
                                                                     valid_data[conf.feature_columns], valid_data['is_click'], \
                                                                     train_data[['sid', 'click_mode', 'transport_mode']], \
                                                                     valid_data[['sid', 'click_mode', 'transport_mode']]
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=conf.cate_columns)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=conf.cate_columns)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=500, num_boost_round=40000, verbose_eval=50)
                              # ,feval=lambda y, t: eval_f1(train_sid_valid, y))
        lgb_model.save_model('../models/model_{}'.format(i))
        val_pred = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
        val_score = score_accurate_rate(train_sid_valid, val_pred)
        f1_score = score_f1(train_sid_valid, val_pred)
        result_proba.append(lgb_model.predict(test_df[conf.feature_columns], num_iteration=lgb_model.best_iteration))
        acc_scores.append(val_score)
        f1_scores.append(f1_score)
        i += 1
    print('cv accurate rate score: ', np.mean(acc_scores))
    print('cv f1  score: ', np.mean(f1_scores))
    pred_test = np.mean(result_proba, axis=0)
    return pred_test


def get_train_test_feats():
    """
    获取训练数据、测试数据
    :return:
    """
    data = pd.read_csv('../data/features/expanded_feature_sample.csv')
    train_df = data[data['click_mode'] != -1]
    test_df = data[data['click_mode'] == -1]
    return train_df, test_df


def get_train_valid_feats():
    """
    获取训练数据、验证数据
    :return:
    """
    data = pd.read_csv('../data/features/expanded_feature_sample.csv')
    train_df = data[data['click_mode'] != -1]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    train_df_t, train_df_e = split_data_by_clickmode(train_df, kfold)
    return train_df_t, train_df_e


def tp_submit():
    """
    训练模型，生成测试集预测结果
    :return:
    """
    train_df, test_df = get_train_test_feats()
    result_lgb = train_lgb(train_df, test_df)
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}.csv'.format('lgb', now_time)
    submit_result(test_df, result_lgb, file_name)


def tp_valid():
    """
    训练模型，生成验证集预测结果
    :return:
    """
    train_df_t, train_df_e = get_train_valid_feats()
    result_lgb = train_lgb(train_df_t, train_df_e)
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}_valid.csv'.format('lgb', now_time)
    test_sid_df = train_df_e[['sid', 'click_mode']]
    test_sid_df['click_mode'] = test_sid_df['click_mode'].astype(np.int32)
    test_sid_df['mode_pro'] = result_lgb
    test_sid_df['max_pro'] = test_sid_df.groupby(['sid'])['mode_pro'].transform(np.max)
    final_submit = test_sid_df.loc[test_sid_df['mode_pro'] == test_sid_df['max_pro'], ['sid', 'transport_mode', 'click_mode']].drop_duplicates(subset=['sid'])
    final_submit = final_submit.rename(columns={'transport_mode': 'recommend_mode'})
    final_submit.to_csv(file_name, index=False)


def predict_by_model():
    """
    预测训练结果
    :return:
    """
    train_df_t, train_df_e = get_train_test_feats()
    result_proba = []
    scores = []
    for i in range(5):
        print('***************************{}'.format(i))
        lgb_model = lgb.Booster(model_file='../models/model_{}'.format(i))
        test_pred = lgb_model.predict(train_df_e[conf.feature_columns], num_iteration=lgb_model.best_iteration)
        val_score = score_accurate_rate(train_df_e[['sid', 'click_mode', 'transport_mode']], test_pred)
        result_proba.append(test_pred)
        scores.append(val_score)

    print('cv accurate_rate-score: ', np.mean(scores))
    pred_test = np.mean(result_proba, axis=0)

    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    file_name = '../submit/{}_result_{}.csv'.format('lgb', now_time)
    submit_result(train_df_e, pred_test, file_name)


if __name__ == '__main__':
    tp_submit()
