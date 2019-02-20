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
            处理方法：删除、保留
        2.同名表
            2.1 有关联任务的
                2.1.1 集市到集市 guldan/10k/hope 
                      2.1.1.1 集市到集市
                         A:同步任务没有名字的(--)：任务类型都是 ugdap grant authorization
                         B:有同步任务名字
                      2.1.1.2 仓库到集市
                      2.1.1.3 集市到集市+仓库到集市
                      找到同步任务核实：10k同名表与evil同名表否来自同一来源（找到两个同步任务，看脚本）
                      处理方法：同一来源保留10k
                               不同一源:更名，修改关联任务脚本，修改同步任务表名
                2.1.2 仓库到集市 10k：保留10k  
            2.2 没有关联任务的与集市负责人沟通
                是同一表，任务被删除：保留10k
                不是同一表，更名，修改关联任务脚本，修改同步任务表名
        3.所有同名表问题解决
          3.1 查询元数据地址，确定需要拷贝的文件和表信息；查询关联的任务？确定有没有写死的地址？ns ip等
              解决代码写死问题；生成地址待拷贝文件列表；生成待修改元数据表脚本
          3.2 拷贝文件
              ns1/ns100-->ns3
          3.3 修改表元数据 location
          3.4 修改汇报主机
          
                
    """
    # 添加文件最后分区日期
    df_all = pd.read_csv('data_all.csv', sep=',', encoding='gbk')
    df_all['last_dt1'] = '9999-99-99'
    df_all.loc[df_all['集市表最后分区'].notnull(), 'last_dt1'] = df_all.loc[df_all['集市表最后分区'].notnull(), '集市表最后分区'].apply(gen_dt)
    df_all['last_dt2'] = '9999-99-99'
    df_all.loc[df_all['已上集市表最后分区'].notnull(), 'last_dt2'] = df_all.loc[df_all['已上集市表最后分区'].notnull(), '已上集市表最后分区'].apply(gen_dt)
    # 两个分区中小的，通过该字段判断是否使老的
    df_all['last_dt'] = df_all[['last_dt1', 'last_dt2']].apply(lambda x: min(x['last_dt1'], x['last_dt2']), axis=1)
    # 是否是同名表
    df_all['is_dup_name'] = 0
    df_all.loc[df_all['源数据集市.1'].notnull(), 'is_dup_name'] = 1
    # 是否关联同步任务 0:没有关联任务 1:关联集市任务  2:关联仓库任务 3:关联两种任务
    df_all['have_task'] = 0
    df_all.loc[df_all['同步任务id'].notnull(), 'have_task'] = 1
    df_all.loc[df_all['仓库至集市任务id'].notnull(), 'have_task'] = 2
    df_all.loc[(df_all['同步任务id'].notnull()) & (df_all['仓库至集市任务id'].notnull()), 'have_task'] = 3

    df_all.to_csv('data_all_flag.csv', sep=',', index=False, encoding='gbk', na_rep='NULL')

    cut_columns = [u'源数据集市', u'NameSpace', u'集市表创建账号', u'集市表库名', u'集市表名', u'集市表最后分区',
                   u'已上集市表创建账号', u'已上集市表最后分区',
                   u'同步任务id', u'同步任务名称', u'同步任务源集群', u'同步任务目标集群', u'同步任务最后执行时间', u'同步任务类型',
                   u'仓库至集市任务id', u'仓库至集市任务名称', u'仓库至集市任务源集市', u'仓库至集市任务类型', u'仓库至集市任务最后执行时间']
    flag_columns = ['last_dt1', 'last_dt2', 'last_dt', 'is_dup_name', 'have_task']
    df_all[cut_columns + flag_columns].to_csv('data_all_flag_cut.csv', sep=',', index=False, encoding='gbk', na_rep='NULL')
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
