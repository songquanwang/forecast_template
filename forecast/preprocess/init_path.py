# coding:utf-8
"""
__file__

    init.py

__description__
   init path


__author__

    songquanwang

"""

import os

import forecast.conf.model_params_conf as config


def init_path():
    # create base feat
    if not os.path.exists(config.solution_feat_base):
        os.makedirs(config.solution_feat_base)
    # create combined feat
    if not os.path.exists(config.solution_feat_combined):
        os.makedirs(config.solution_feat_combined)
    # create data
    if not os.path.exists(config.solution_data):
        os.makedirs(config.solution_data)
    # create output
    if not os.path.exists(config.solution_output):
        os.makedirs(config.solution_output)
    # create info
    if not os.path.exists(config.solution_info):
        os.makedirs(config.solution_info)

    # creat folder for the training and testing feat
    if not os.path.exists("%s/All" % config.solution_feat_base):
        os.makedirs("%s/All" % config.solution_feat_base)

    if not os.path.exists("%s/All" % config.solution_feat_combined):
        os.makedirs("%s/All" % config.solution_feat_combined)

    # creat folder for each run and fold
    for run in range(1, config.n_runs + 1):
        for fold in range(1, config.n_folds + 1):
            path_base = "%s/Run%d/Fold%d" % (config.solution_feat_base, run, fold)
            path_combined = "%s/Run%d/Fold%d" % (config.solution_feat_combined, run, fold)
            if not os.path.exists(path_base):
                os.makedirs(path_base)
            if not os.path.exists(path_combined):
                os.makedirs(path_combined)


if __name__ == "__main__":
    init_path()
