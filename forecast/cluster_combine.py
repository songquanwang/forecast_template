import numpy as np
import re
import pandas as pd


def gen_dt(x):
    """
    获取分区日期
    注意：没有dt=****-**-**的，默认分区9999-99-99 代表最新
    s='dt=2019-02-13/day_num=7'
    :param x:
    :return:
    """
    date_all = re.findall(r"(?<=\bdt\=)\d{4}-\d{1,2}-\d{1,2}", x)
    if len(date_all) > 0:
        return date_all[0]
    return '9999-99-99'


def get_old(df, date_str):
    """
    获取陈旧表
    :param df:
    :param date_str:
    :return:
    """
    return df[(df['last_dt1'] < date_str) | ((df['last_dt2'] < date_str))]


def get_dupname(df):
    """
    获取同名表
    等价于：
        df_all[df_all['集市表名']==df_all['已上集市表名']]
    :param df:
    :return:
    """
    return df[df['源数据集市.1'].notnull()]


def get_no_task(df):
    """
    获取没有同步任务表
    :param df:
    :return:
    """
    return df[(df['同步任务id'].isnull()) & (df['仓库至集市任务id'].isnull())]


def get_task(df):
    """
    获取有同步任务表
    :param df:
    :return:
    """
    return df[(df['同步任务id'].notnull()) | (df['仓库至集市任务id'].notnull())]


if __name__ == "__main__":
    """
        1.old 表需要集市管理负责人沟通删除
            1.1 evil 老 10k 老
            1.2 evil 老 10k 新
            1.3 evil 新 10k 老
            1.4 非重名表
        2.同名表中非old部分
            2.1 有关联任务的可以认定为相同
                2.1.1 集市到集市 guldan/10k/hope 
                     找到同步任务核实：10k同名表是否来自同一来源
                2.1.2 仓库到集市 10k：保留10k  
            2.2 没有关联任务的与集市负责人沟通
                
    """
    df_all = pd.read_csv('BGS_SZ_20190214.csv', sep=',', encoding='gbk')
    df_all['last_dt1'] = '9999-99-99'
    df_all.loc[df_all['集市表最后分区'].notnull(), 'last_dt1'] = df_all.loc[df_all['集市表最后分区'].notnull(), '集市表最后分区'].apply(gen_dt)
    df_all['last_dt2'] = '9999-99-99'
    df_all.loc[df_all['已上集市表最后分区'].notnull(), 'last_dt2'] = df_all.loc[df_all['已上集市表最后分区'].notnull(), '已上集市表最后分区'].apply(gen_dt)
    df_all.to_csv('data_all.csv', sep=',', index=False, encoding='gbk', na_rep='NULL')
    # old 表 需要集市 719
    df_old = get_old(df_all, '2018-08-19')

    # 同名表 1315
    df_dupname = get_dupname(df_all)
    # 重名表中非old部分 1133
    df_dupname_new = df_dupname.loc[~df_dupname.index.isin(df_old.index)]
    # 182
    df_dupname_old = df_dupname.loc[df_dupname.index.isin(df_old.index)]
    # 同名表 非 old 有关联任务 710
    df_dupname_new_task = get_task(df_dupname_new)
    # 423
    df_dupname_new_notask = get_no_task(df_dupname_new)
