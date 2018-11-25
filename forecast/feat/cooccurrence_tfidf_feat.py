# coding:utf-8
"""
__file__

    genFeat_cooccurrence_tfidf.py
    1.把query title description 三个列生成 一元、二元、三元 9个特征
    2.两两特征词根数组笛卡尔积，生成新特征
__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. tfidf for the following cooccurrence terms
            - query unigram/bigram & title unigram/bigram
            - query unigram/bigram & description unigram/bigram
            - query id & title unigram/bigram
            - query id & description unigram/bigram

        2. corresponding lsa (svd) version features

__author__

    songquanwang

"""

import _pickle as pickle
import abc

from sklearn.decomposition import TruncatedSVD

from forecast.feat.nlp.nlp_utils import getTFV
import forecast.conf.model_params_conf as config
from forecast.feat.abstract_base_feat import AbstractBaseFeat


class CooccurenceTfidfFeat(AbstractBaseFeat):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        多个字段里 不同单词组合在一起作为一个整体
        query title    11 12  21 22
        query des      11 12  21 22
        query_id title u  b
        query_id dex   u  b

        """
        self.column_names = [
            "query_unigram_title_unigram",
            "query_unigram_title_bigram",
            "query_unigram_description_unigram",
            "query_unigram_description_bigram",

            "query_bigram_title_unigram",
            "query_bigram_title_bigram",
            "query_bigram_description_unigram",
            "query_bigram_description_bigram",

            "query_id_title_unigram",
            "query_id_title_bigram",
            "query_id_description_unigram",
            "query_id_description_bigram",
        ]

        self.ngram_range = config.cooccurrence_tfidf_ngram_range

        self.svd_n_components = 100

    @staticmethod
    def cooccurrence_terms(lst1, lst2, join_str):
        """
            Cooccurrence terms：两组单词任意组合，用join_str链接
        :param lst1:
        :param lst2:
        :param join_str: len(lst1)*len(lst2) 长度的单词组合[],用空格join成一个字符串返回
        :return:
        """
        terms = [""] * len(lst1) * len(lst2)
        cnt = 0
        for item1 in lst1:
            for item2 in lst2:
                terms[cnt] = item1 + join_str + item2
                cnt += 1
        res = " ".join(terms)
        return res

    def gen_cooccurrence_column(self, df):
        cooccurrence_terms = CooccurenceTfidfFeat.cooccurrence_terms
        # cooccurrence terms
        join_str = "X"
        # query_unigram * [titile,description] =4
        df["query_unigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_unigram"], join_str), axis=1))
        df["query_unigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_bigram"], join_str), axis=1))
        df["query_unigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_unigram"], join_str), axis=1))
        df["query_unigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_bigram"], join_str), axis=1))
        # query_bigram * [titile,description] =4
        df["query_bigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_unigram"], join_str), axis=1))
        df["query_bigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_bigram"], join_str), axis=1))
        df["query_bigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_unigram"], join_str), axis=1))
        df["query_bigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_bigram"], join_str), axis=1))
        # query_id * [titile,description] =4
        df["query_id_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["title_unigram"], join_str), axis=1))
        df["query_id_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["title_bigram"], join_str), axis=1))
        df["query_id_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["description_unigram"], join_str), axis=1))
        df["query_id_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid" + str(x["qid"])], x["description_bigram"], join_str), axis=1))

    def gen_feat(self, path, dfTrain, dfTest, mode, feat_names):
        """
        只提取feat_names这些特征
        :param dfTrain:
        :param dfTest:
        :param mode:
        :param ngram_range:
        :param feat_names:
        :param column_names:
        :return:
        """
        new_feat_names = []
        for feat_name, column_name in zip(feat_names, self.column_names):
            print("generate %s feat" % feat_name)
            # tfidf
            tfv = getTFV(ngram_range=self.ngram_range)
            # 共6906个单词的字典
            X_tfidf_train = tfv.fit_transform(dfTrain[column_name])
            X_tfidf_test = tfv.transform(dfTest[column_name])

            with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                pickle.dump(X_tfidf_train, f, -1)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
                pickle.dump(X_tfidf_test, f, -1)

            # svd 提取100个主成分
            svd = TruncatedSVD(n_components=self.svd_n_components, n_iter=15)
            X_svd_train = svd.fit_transform(X_tfidf_train)
            X_svd_test = svd.transform(X_tfidf_test)
            with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, self.svd_n_components), "wb") as f:
                pickle.dump(X_svd_train, f, -1)
            with open("%s/%s.%s_individual_svd%d.feat.pkl" % (path, mode, feat_name, self.svd_n_components), "wb") as f:
                pickle.dump(X_svd_test, f, -1)

    def gen_feat_cv(self):
        """
        cooccurrence terms column names
        共24个特征 tfidf tfidf_individual_svd 各12个
        :return:
        """
        # feature names
        feat_names = [name + "_tfidf" for name in self.column_names]

        svd_n_components = 100

        # Load Data
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = pickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            dfTest = pickle.load(f)
        # load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, config.stratified_label), "rb") as f:
            skf = pickle.load(f)

        print("==================================================")
        print("Generate co-occurrence tfidf features...")

        # gen temp feat
        self.gen_column_gram(dfTrain)
        self.gen_column_gram(dfTest)
        # get cooccurrence terms
        self.gen_cooccurrence_column(dfTrain)
        self.gen_cooccurrence_column(dfTest)

        # Cross validation
        print("For cross-validation...")
        for run in range(config.n_runs):
            # use 33% for training and 67 % for validation ,so we switch trainInd and validInd
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                path = "%s/Run%d/Fold%d" % (config.solution_feat_base, run + 1, fold + 1)
                X_tfidf_train = dfTrain.iloc[trainInd]
                X_tfidf_valid = dfTrain.iloc[validInd]
                self.gen_feat(path, X_tfidf_train, X_tfidf_valid, "valid", feat_names)

        print("Done.")

        # Re-training
        print("For training and testing...")
        path = "%s/All" % config.solution_feat_base
        self.gen_feat(path, dfTrain, dfTest, "test", feat_names)
        print("Done.")

        # 记录所有的特征
        new_feat_names = []
        new_feat_names.extend(feat_names)
        new_feat_names += ["%s_individual_svd%d" % (f, svd_n_components) for f in feat_names]

        # 保存所有的特征名字：intersect_tfidf.feat_name
        feat_name_file = "%s/cooccurrence_tfidf.feat_name" % config.solution_feat_combined
        print("Feature names are stored in %s" % feat_name_file)
        self.dump_feat_name(new_feat_names, feat_name_file)

        print("All Done.")


if __name__ == "__main__":
    cooccurence_tfidf_feat = CooccurenceTfidfFeat()
    cooccurence_tfidf_feat.gen_feat_cv()
