# coding:utf-8
"""
__file__

    preprocess.py

__description__

    This file preprocesses data.
    train 文件格式："id","query","product_title","product_description","median_relevance","relevance_variance"
    test  文件格式："id","query","product_title","product_description"

__author__

    songquanwang
    
"""

import _pickle as  pickle

import numpy as np
import pandas as pd

from forecast.feat.nlp.nlp_utils import clean_text
import forecast.conf.model_params_conf as config


def preprocess():
    """
    dolumns =['id', 'query', 'product_title', 'product_description',
       'median_relevance', 'relevance_variance',
        'index', 'median_relevance_1','median_relevance_2', 'median_relevance_3', 'median_relevance_4',
        'qid']
    id: 唯一编号 Train Test 1-32671  train test 混合在一起
    dfTrain:10158
    dfTest:22513
    1.load  train and test data
    2.add index: 从0 开始编号
    3.dummy median_relevance：生成median_relevance_1 median_relevance_2 median_relevance_3 median_relevance_4
    4.add qid  :query distinct 后的序号
    5.替换一些同义词，清除html标记
    """

    print("Load data...")

    dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
    dfTest = pd.read_csv(config.original_test_data_path).fillna("")
    # number of train/test samples
    num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

    print("Done.")

    ######################
    ## Pre-process Data ##
    ######################
    print("Pre-process data...")

    # median_relevance=1  relevance_variance=0
    dfTest["median_relevance"] = np.ones((num_test))
    dfTest["relevance_variance"] = np.zeros((num_test))

    # 从0 开始编号
    dfTrain["index"] = np.arange(num_train)
    dfTest["index"] = np.arange(num_test)

    # one-hot encode the median_relevance ：dummy median_relevance
    for i in range(config.num_of_class):
        dfTrain["median_relevance_%d" % (i + 1)] = 0
        dfTrain["median_relevance_%d" % (i + 1)][dfTrain["median_relevance"] == (i + 1)] = 1

    # query ids 从1开始编号,按照从上到下顺序； query train test 具有相同的集合
    qid_dict = dict()
    for i, q in enumerate(np.unique(dfTrain["query"]), start=1):
        qid_dict[q] = i

    # query id 1-261 个唯一的查询（train test)都相同
    dfTrain["qid"] = list(map(lambda q: qid_dict[q], dfTrain["query"]))
    dfTest["qid"] = list(map(lambda q: qid_dict[q], dfTest["query"]))

    # clean text : query title description
    clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
    # axis =1 传入每行(多个列)
    dfTrain = dfTrain.apply(clean, axis=1)
    dfTest = dfTest.apply(clean, axis=1)

    print("Done.")

    ###############
    ## Save Data ##
    ###############
    print("Save data...")

    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)

    print("Done.")


if __name__ == "__main__":
    preprocess()
