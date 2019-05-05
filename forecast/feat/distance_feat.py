# coding:utf-8
"""
__file__

    genFeat_distance_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. jaccard coefficient/dice distance between query & title, query & description, title & description pairs
            - just plain jaccard coefficient/dice distance
            - compute for unigram/bigram/trigram

        2. jaccard coefficient/dice distance stats features for title/description
            - computation is carried out with regard to a pool of samples grouped by:
                - median_relevance (#4)
                - query (qid) & median_relevance (#4)
            - jaccard coefficient/dice distance for the following pairs are computed for each sample
                - sample title        vs.  pooled sample titles
                - sample description  vs.  pooled sample descriptions
                Note that in the pool samples, we exclude the current sample being considered.
            - stats features include quantiles of cosine similarity and others defined in the variable "stats_func", e.g.,
                - mean value
                - standard deviation (std)
                - more can be added, e.g., moment features etc

__author__

    songquanwang

"""

import _pickle as pickle
import abc

import numpy as np

import forecast.conf.model_params_conf as  config
from forecast.feat.basic_tfidf_feat import AbstractBaseFeat
import forecast.utils.distance_utils as utils

import forecast.conf.feat_params_conf as feat_params_conf


class DistanceFeat(AbstractBaseFeat):
    __metaclass__ = abc.ABCMeta

    def __init__(self):

        # stats to extract
        self.quantiles_range = np.arange(0, 1.5, 0.5)
        self.stats_func = [np.mean, np.std]
        self.stats_feat_num = len(self.quantiles_range) + len(self.stats_func)

    @staticmethod
    def extract_basic_distance_feat(df):
        """
        计算 query title description的 jaccard dice距离
        :param df:
        :return:
        """
        # jaccard coef/dice dist of n-gram
        print("generate jaccard coef and dice dist for n-gram")
        dists = ["jaccard_coef", "dice_coef"]
        grams = ["unigram", "bigram", "trigram"]
        # 计算 query title description的 jaccard dice距离
        feat_names = ["query", "title", "description"]
        for dist in dists:
            for gram in grams:
                for i in range(len(feat_names) - 1):
                    for j in range(i + 1, len(feat_names)):
                        target_name = feat_names[i]
                        obs_name = feat_names[j]
                        df["%s_of_%s_between_%s_%s" % (dist, gram, target_name, obs_name)] = list(
                            df.apply(lambda x: utils.compute_jaccard_dice_dist(x[target_name + "_" + gram], x[obs_name + "_" + gram], dist), axis=1))

    def extract_statistical_distance_feat(self, path, dfTrain, dfTest, mode):
        """
        计算 title description 的 一元 二元 三元 统计信息 (使用 jaccard_coef，dice_coef)两种距离方式
        :param path:
        :param dfTrain:
        :param dfTest:
        :param mode:
        :param feat_names:
        :return:
        """
        new_feat_names = []
        ## get the indices of pooled samples
        relevance_indices_dict = self.get_sample_indices_by_relevance(dfTrain)
        query_relevance_indices_dict = self.get_sample_indices_by_relevance(dfTrain, "qid")
        # very time consuming
        # 只计算title description 没有计算query
        for dist in ["jaccard_coef", "dice_coef"]:
            for name in ["title", "description"]:
                for gram in ["unigram", "bigram", "trigram"]:
                    # title 的 一元 二元 三元 计算统计信息
                    dist_stats_feat_by_relevance_train = self.generate_dist_stats_feat(dist, dfTrain[name + "_" + gram].values, dfTrain["id"].values,
                                                                                       dfTrain[name + "_" + gram].values, dfTrain["id"].values,
                                                                                       relevance_indices_dict)
                    dist_stats_feat_by_query_relevance_train = self.generate_dist_stats_feat(dist, dfTrain[name + "_" + gram].values, dfTrain["id"].values,
                                                                                             dfTrain[name + "_" + gram].values, dfTrain["id"].values,
                                                                                             query_relevance_indices_dict,
                                                                                             dfTrain["qid"].values)
                    with open("%s/train.%s_%s_%s_stats_feat_by_relevance.feat.pkl" % (path, name, gram, dist), "wb") as f:
                        pickle.dump(dist_stats_feat_by_relevance_train, f, -1)
                    with open("%s/train.%s_%s_%s_stats_feat_by_query_relevance.feat.pkl" % (path, name, gram, dist),
                              "wb") as f:
                        pickle.dump(dist_stats_feat_by_query_relevance_train, f, -1)
                    ## test
                    dist_stats_feat_by_relevance_test = self.generate_dist_stats_feat(dist, dfTrain[name + "_" + gram].values, dfTrain["id"].values,
                                                                                      dfTest[name + "_" + gram].values, dfTest["id"].values,
                                                                                      relevance_indices_dict)
                    dist_stats_feat_by_query_relevance_test = self.generate_dist_stats_feat(dist, dfTrain[name + "_" + gram].values, dfTrain["id"].values,
                                                                                            dfTest[name + "_" + gram].values, dfTest["id"].values,
                                                                                            query_relevance_indices_dict,
                                                                                            dfTest["qid"].values)
                    with open("%s/%s.%s_%s_%s_stats_feat_by_relevance.feat.pkl" % (path, mode, name, gram, dist),
                              "wb") as f:
                        pickle.dump(dist_stats_feat_by_relevance_test, f, -1)
                    with open("%s/%s.%s_%s_%s_stats_feat_by_query_relevance.feat.pkl" % (path, mode, name, gram, dist),
                              "wb") as f:
                        pickle.dump(dist_stats_feat_by_query_relevance_test, f, -1)

                    # update feat names
                    new_feat_names.append("%s_%s_%s_stats_feat_by_relevance" % (name, gram, dist))
                    new_feat_names.append("%s_%s_%s_stats_feat_by_query_relevance" % (name, gram, dist))


        return new_feat_names

    def gen_feat(self, path, dfTrain, dfTest, mode, feat_names):
        new_feat_names = []
        for feat_name in feat_names:
            X_train = dfTrain[feat_name]
            X_test = dfTest[feat_name]
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                pickle.dump(X_train, f, -1)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
                pickle.dump(X_test, f, -1)
            # add basic feat
            new_feat_names.append(feat_name)
            # extract statistical distance features
            if feat_params_conf.stats_feat_flag:
                dfTrain_copy = dfTrain.copy()
                dfTest_copy = dfTest.copy()
                added_feat_names = self.extract_statistical_distance_feat(path, dfTrain_copy, dfTest_copy, mode, feat_names)
                new_feat_names.extend(added_feat_names)
        return new_feat_names

    def gen_feat_cv(self):
        """

        :return:
        """

        # Load Data
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = pickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            dfTest = pickle.load(f)
        ## load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, config.stratified_label), "rb") as f:
            skf = pickle.load(f)

        #######################
        ## Generate Features ##
        #######################
        print("==================================================")
        print("Generate distance features...")
        # add column gram
        self.gen_column_gram(dfTrain)
        self.gen_column_gram(dfTest)
        # add basic distince feat
        DistanceFeat.extract_basic_distance_feat(dfTrain)
        DistanceFeat.extract_basic_distance_feat(dfTest)

        feat_names = [name for name in dfTrain.columns if "jaccard_coef" in name or "dice_coef" in name]

        print("For cross-validation...")
        for run in range(config.n_runs):
            # use 33% for training and 67 % for validation,so we switch trainInd and validInd
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                path = "%s/Run%d/Fold%d" % (config.solution_feat_base, run + 1, fold + 1)

                X_train_train = dfTrain.iloc[trainInd]
                X_train_valid = dfTrain.iloc[validInd]
                self.gen_feat(path, X_train_train, X_train_valid, "valid", feat_names)

        print("Done.")

        print("For training and testing...")
        path = "%s/All" % config.solution_feat_base

        added_feat_names = self.gen_feat(path, dfTrain, dfTest, "test", feat_names)

        # 保存所有的特征名字 ：distance.feat_name
        new_feat_names = []
        new_feat_names.extend(added_feat_names)
        feat_name_file = "%s/distance.feat_name" % config.solution_feat_combined
        print("Feature names are stored in %s" % feat_name_file)
        self.dump_feat_name(new_feat_names, feat_name_file)

        print("All Done.")


if __name__ == "__main__":
    distance_feat = DistanceFeat()
    distance_feat.gen_feat_cv()
