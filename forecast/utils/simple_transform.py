# coding:utf-8
"""
__file__

	feat_utils.py

__description__

	This file provides utils for generating features.

__author__

	songquanwang

"""
from sklearn.base import BaseEstimator


def identity(x):
    return x


class SimpleTransform(BaseEstimator):
    """
      adopted from @Ben Hamner's Python Benchmark code
      https://www.kaggle.com/benhamner/crowdflower-search-relevance/python-benchmark
    """

    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)
