# coding:utf-8
"""
__file__

    genFeat_counting_feat.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. Basic Counting Features
            
            1. Count of n-gram in query/title/description

            2. Count & Ratio of Digit in query/title/description

            3. Count & Ratio of Unique n-gram in query/title/description

        2. Intersect Counting Features

            1. Count & Ratio of a's n-gram in b's n-gram

        3. Intersect Position Features

            1. Statistics of Positions of a's n-gram in b's n-gram

            2. Statistics of Normalized Positions of a's n-gram in b's n-gram

__author__

    songquanwang

"""

import _pickle as pickle
import abc

import numpy as np

import forecast.conf.model_params_conf as  config
import forecast.utils.utils as utils
from forecast.feat.abstract_base_feat import AbstractBaseFeat


class CountingFeat(AbstractBaseFeat):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_position_list(target, obs):
        """
            Get the list of positions of obs in target
            从obs中找到（在target存在的词）的index（index从1开始）
            如果obs中没有target中的元素，返回[0]
        """
        pos_of_obs_in_target = [0]
        if len(obs) != 0:
            # 如果 obs 中元素在target中，保留obs的序号
            pos_of_obs_in_target = [j for j, w in enumerate(obs, start=1) if w in target]
            if len(pos_of_obs_in_target) == 0:
                pos_of_obs_in_target = [0]
        return pos_of_obs_in_target

    @staticmethod
    def extract_digit_count_feat(df, feat_names, grams):
        """
         word count and digit count 22个 +1
        :param df:
        :param feat_names:
        :param grams:
        :return:
        """
        print("generate word counting features")
        # 计算包含数字的个数
        count_digit = lambda x: sum([1. for w in x if w.isdigit()])
        for feat_name in feat_names:
            for gram in grams:
                # 单词个数
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                # 唯一单词个数
                df["count_of_unique_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                # 唯一单词占比
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = list(map(utils.try_divide, df["count_of_unique_%s_%s" % (feat_name, gram)], df["count_of_%s_%s" % (feat_name, gram)]))

            # unigram中数值个数
            df["count_of_digit_in_%s" % feat_name] = list(df.apply(lambda x: count_digit(x[feat_name + "_unigram"]), axis=1))
            # unigram中数值个数/unigram中所有单词数量
            df["ratio_of_digit_in_%s" % feat_name] = list(map(utils.try_divide, df["count_of_digit_in_%s" % feat_name], df["count_of_%s_unigram" % (feat_name)]))

        # 1 没有描述信息 ；0 有描述信息
        df["description_missing"] = list(df.apply(lambda x: int(x["description_unigram"] == ""), axis=1))

    @staticmethod
    def extract_interset_digit_count_feat(df, feat_names, grams):
        """
        intersect word count 48个
        query title des-->query title ;query des;title des
        :param df:
        :param feat_names:
        :param grams:
        :return:
        """
        print("generate intersect word counting features")
        # unigram
        for gram in grams:
            for obs_name in feat_names:
                for target_name in feat_names:
                    if target_name != obs_name:
                        # 两个不同字段中交集的词语个；因为有重复词语，A&B ！=B&A

                        df["count_of_%s_%s_in_%s" % (obs_name, gram, target_name)] = list(df.apply(lambda x: sum([1. for w in x[obs_name + "_" + gram] if w in set(x[target_name + "_" + gram])]), axis=1))
                        # 交集占单词个数比例
                        df["ratio_of_%s_%s_in_%s" % (obs_name, gram, target_name)] = list(map(utils.try_divide, df["count_of_%s_%s_in_%s" % (obs_name, gram, target_name)],df["count_of_%s_%s" % (obs_name, gram)]))

            # title_query_count/query_count
            df["title_%s_in_query_div_query_%s" % (gram, gram)] = list(map(utils.try_divide, df["count_of_title_%s_in_query" % gram], df["count_of_query_%s" % gram]))
            # title
            df["title_%s_in_query_div_query_%s_in_title" % (gram, gram)] = list(map(utils.try_divide, df["count_of_title_%s_in_query" % gram], df["count_of_query_%s_in_title" % gram]))
            df["description_%s_in_query_div_query_%s" % (gram, gram)] = list(map(utils.try_divide, df["count_of_description_%s_in_query" % gram], df["count_of_query_%s" % gram]))
            df["description_%s_in_query_div_query_%s_in_description" % (gram, gram)] = list(map(utils.try_divide, df["count_of_description_%s_in_query" % gram],
                                                                                           df["count_of_query_%s_in_description" % gram]))


    @staticmethod
    def extract_interset_word_pos_feat(df, feat_names, grams):
        """
        intersect word position feat
        一个字段的词根，在另个字段的词根 的位置[最小、最大、中位数、平均值、标准差]
        :param df:
        :param feat_names:
        :param grams:
        :return:
        """
        print("generate intersect word position features")
        for gram in grams:
            for target_name in feat_names:
                for obs_name in feat_names:
                    if target_name != obs_name:
                        pos = list(df.apply(lambda x: CountingFeat.get_position_list(x[target_name + "_" + gram], obs=x[obs_name + "_" + gram]), axis=1))
                        ## stats feat on pos
                        df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = list(map(np.min, pos))
                        df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = list(map(np.mean, pos))
                        df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] =list(map(np.median, pos))
                        df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = list(map(np.max, pos))
                        df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = list(map(np.std, pos))
                        # stats feat on normalized_pos
                        df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = list(map(utils.try_divide,
                                                                                                      df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)],
                                                                                                      df["count_of_%s_%s" % (obs_name, gram)]))
                        df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = list(map(utils.try_divide,
                                                                                                       df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)],
                                                                                                       df["count_of_%s_%s" % (obs_name, gram)]))
                        df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = list(map(utils.try_divide,
                                                                                                         df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)],
                                                                                                         df["count_of_%s_%s" % (obs_name, gram)]))
                        df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = list(map(utils.try_divide,
                                                                                                      df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)],
                                                                                                      df["count_of_%s_%s" % (obs_name, gram)]))
                        df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = list(map(utils.try_divide,
                                                                                                      df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)],
                                                                                                      df["count_of_%s_%s" % (obs_name, gram)]))

    def extract_feat(self, df):
        """
        1.word count and digit count
        2.intersect word count
        3.intersect word position feat
        :param df:
        :param feat_names:
        :param grams:
        :return:
        """
        # 生成临时特征
        feat_names = ["query", "title", "description"]
        grams = ["unigram", "bigram", "trigram"]
        # word count and digit count
        print("generate word counting features")
        # 计算包含数字的个数
        CountingFeat.extract_digit_count_feat(df, feat_names, grams)
        # intersect word count
        print("generate intersect word counting features")
        CountingFeat.extract_interset_digit_count_feat(df, feat_names, grams)
        # intersect word position feat
        print("generate intersect word position features")
        CountingFeat.extract_interset_word_pos_feat(df, feat_names, grams)

    def gen_feat(self, path, dfTrain, dfTest, mode, feat_names):
        """
        只提取feat_names这些特征
        :param dfTrain:
        :param dfTest:
        :param mode:
        :param feat_names:
        :return:
        """
        for feat_name in feat_names:
            X_train = dfTrain[feat_name]
            X_test = dfTest[feat_name]
            with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                pickle.dump(X_train, f, -1)
            with open("%s/%s.%s.feat.pkl" % (path, mode, feat_name), "wb") as f:
                pickle.dump(X_test, f, -1)

    def gen_feat_cv(self):
        """

        :return:
        """
        with open(config.processed_train_data_path, "rb") as f:
            dfTrain = pickle.load(f)
        with open(config.processed_test_data_path, "rb") as f:
            dfTest = pickle.load(f)
        # load pre-defined stratified k-fold index
        with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, config.stratified_label), "rb") as f:
            skf = pickle.load(f)

        print("==================================================")
        print("Generate counting features...")

        # 生成临时特征
        self.gen_column_gram(dfTrain)
        self.gen_column_gram(dfTest)
        # 生成其他特征
        self.extract_feat(dfTrain)
        self.extract_feat(dfTest)

        feat_names = [
            name for name in dfTrain.columns  if "count" in name  or "ratio" in name or "div" in name or "pos_of" in name
        ]
        feat_names.append("description_missing")

        print("For cross-validation...")
        for run in range(config.n_runs):
            # use 33% for training and 67 % for validation, so we switch trainInd and validInd
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("Run: %d, Fold: %d" % (run + 1, fold + 1))
                path = "%s/Run%d/Fold%d" % (config.solution_feat_base, run + 1, fold + 1)
                X_train_train = dfTrain.iloc[trainInd]
                X_train_valid = dfTrain.iloc[validInd]
                self.gen_feat(path, X_train_train, X_train_valid, "valid", feat_names)
        print("Done.")

        print("For training and testing...")
        path = "%s/All" % config.solution_feat_base
        # use full version for X_train
        self.gen_feat(path, dfTrain, dfTest, "test", feat_names)

        # 保存所有的特征名字：counting.feat_name
        new_feat_names = feat_names
        feat_name_file = "%s/counting.feat_name" % config.solution_feat_combined
        print("Feature names are stored in %s" % feat_name_file)
        self.dump_feat_name(new_feat_names, feat_name_file)

        print("All Done.")

if __name__ == "__main__":
    counting_feat = CountingFeat()
    counting_feat.gen_feat_cv()
