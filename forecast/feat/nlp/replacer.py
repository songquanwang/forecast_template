# coding:utf-8
"""
    This file provides functions to perform synonym & antonym replacement.
    Such functions are adopted from "Python Text Processing with NLTK 2.0 Cookbook," Chapter 2, Page 39-43.

    Wordnet是一个由普林斯顿大学认识科学实验室在心理学教授乔治·A·米勒的指导下建立和维护的大型的英语词典。
    WordNet的开发有两个目的：
　　1.它既是一个字典，又是一个辞典，它比单纯的辞典或词典都更加易于使用。
　　2.支持自动的文本分析以及人工智能应用。
　　    在WordNet中，名词，动词，形容词和副词各自被组织成一个同义词的网络，每个同义词集合都代表一个基本的语义概念，并且这些集合之间也
    由各种关系连接。（一个多义词将出现在它的每个意思的同义词集合中）。
"""

import csv

from nltk.corpus import wordnet


##########################
## Synonym Replacement ##
##########################
class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return [self.word_map.get(w, w) for w in word]


class CsvWordReplacer(WordReplacer):
    """
    读取同义词表，#为注释
    """

    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            if word.startswith("#"):
                continue
            word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)


##########################
## Antonym Replacement ##
##########################
class AntonymReplacer(object):
    """
    反义词
    Wordnet是一个词典。每个词语(word)可能有多个不同的语义，对应不同的sense。
    而每个不同的语义（sense）又可能对应多个词，如topic和subject在某些情况下是同义的，
    一个sense中的多个消除了多义性的词语叫做lemma。例如，“publish”是一个word，它可能有多个sense：
    一个synset(同义词集：指意义相同的词条的集合)被一个三元组描述：（单词.词性.序号）
    """

    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            # 同义词集中的所有词条
            for lemma in syn.lemmas:
                # 反义词
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name)
        # 刚好有一个反义词才替换
        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None

    def replace_negations(self, sent):
        """
        not word 转成反义词
        :param sent:
        :return:
        """
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i + 1 < l:
                ant = self.replace(sent[i + 1])
                # 有反义词就替换
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words
