# -*- coding: utf-8 -*-
import numpy as np

num_of_class = 4
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size = 1

ebc_hard_threshold = False
verbose_level = 1

# 模型算法包路径 libfm rgf 必须\ 用window下用/不行
libfm_exe = ".\libfm-1.40.windows\libfm.exe"
call_exe = "./rgf1.2/test/call_exe.pl"
rgf_exe = "./rgf1.2/bin/rgf.exe"

## cv交叉验证配置
n_runs = 3
n_folds = 3
stratified_label = "query"

# 路径配置
data_folder = "./data"
solution_folder = "./outputs/solution"
solution_feat = "%s/Feat" % solution_folder
solution_output = "%s/Output" % solution_folder
solution_data = "%s/Data" % solution_folder
solution_info = "%s/Info" % solution_folder
# 基本特征
solution_feat_base = "%s/base" % solution_feat

feat_folder = solution_feat
original_train_data_path = "%s/train.csv" % data_folder
original_test_data_path = "%s/test.csv" % data_folder
processed_train_data_path = "%s/train.processed.csv.pkl" % solution_data
processed_test_data_path = "%s/test.processed.csv.pkl" % solution_data
# 现在没用到，被注释掉了
pos_tagged_train_data_path = "%s/train.pos_tagged.csv.pkl" % solution_data
pos_tagged_test_data_path = "%s/test.pos_tagged.csv.pkl" % solution_data

output_path = solution_output

# nlp related
drop_html_flag = True
basic_tfidf_ngram_range = (1, 3)
basic_tfidf_vocabulary_type = "common"
cooccurrence_tfidf_ngram_range = (1, 1)
cooccurrence_word_exclude_stopword = False
stemmer_type = "porter"  # "snowball"

# transform for count features
count_feat_transform = np.sqrt
# try 10/50/100
ensemble_model_top__k = 10
