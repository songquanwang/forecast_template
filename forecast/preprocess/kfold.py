# coding:utf-8
"""
__file__

    init.py

__description__
   init path


__author__

    songquanwang

"""

import _pickle as pickle

from sklearn.cross_validation import StratifiedKFold

import forecast.conf.model_params_conf as config


def gen_stratified_kfold():
    """
     This file generates the StratifiedKFold indices which will be kept fixed in
     ALL the following model building parts.
     分层抽取: 根据median_relevance 也就是 0 1 2 3 各种等级抽取近似；qid 不同的关键字抽取近似
     [
         [[validInd_fold1,trainInd_fold1],[validInd_fold1,trainInd_fold1],[validInd_fold1,trainInd_fold1]],
         [run2],
         [run3]
     ]
    """

    # load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)

    skf = [0] * config.n_runs
    for stratified_label, key in zip(["relevance", "query"], ["median_relevance", "qid"]):
        for run in range(config.n_runs):
            random_seed = 2015 + 1000 * (run + 1)
            skf[run] = StratifiedKFold(dfTrain[key], n_folds=config.n_folds, shuffle=True, random_state=random_seed)
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("================================")
                print("Index for run: %s, fold: %s" % (run + 1, fold + 1))
                print("Train (num = %s)" % len(trainInd))
                print(trainInd[:10])
                print("Valid (num = %s)" % len(validInd))
                print(validInd[:10])
        with open("%s/stratifiedKFold.%s.pkl" % (config.solution_data, stratified_label), "wb") as f:
            pickle.dump(skf, f, -1)


if __name__ == "__main__":
    gen_stratified_kfold()
