# coding:utf-8
"""
nitk :用于英文 分词、词干化
jieba:用户中文分词
BeautifulSoup ：网络爬取、分析的工具箱
    This file provides functions to perform NLP task, e.g., TF-IDF and POS tagging.
    TF-IDF:
    BOW : Bag-of-words model 词袋模型
__author__
    songquanwang

"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from bs4 import BeautifulSoup
from nltk import pos_tag

from forecast.feat.nlp.replacer import CsvWordReplacer
import forecast.conf.model_params_conf as config

"""
    获取stop words 集合
"""
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)

"""
    提取词干：Stemming
    Stemming:是抽取词的词干或词根形式（不一定能够表达完整语义）。
    NLTK中提供了三种最常用的词干提取器接口，
    即 Porter stemmer, Lancaster Stemmer 和 Snowball Stemmer。
"""
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')


def stem_tokens(tokens, stemmer):
    """
    把所有单词转成词干
    :param tokens:
    :param stemmer:
    :return:
    """
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


#############
## POS Tag ##
#############
#
"""
    (?u)就是启用Unicode dependent特性；用于匹配带汉字的文本
    s='abc\tdef' # 字符串中被当成转义字符
    s1=r'abc\tdef' # 把字符串当成普通字符处理
"""
token_pattern = r"(?u)\b\w\w+\b"


def pos_tag_text(line, token_pattern=token_pattern, exclude_stopword=config.cooccurrence_word_exclude_stopword, encode_digit=False):
    """
    词语成分标注
    findall：分词
    stem_tokens：提取词根
    pos_tag ：标注词在文本中的成分
    :param line:
    :param token_pattern:
    :param exclude_stopword:是否去除停用词
    :param encode_digit:
    :return:
    """
    token_pattern = re.compile(token_pattern, flags=re.UNICODE | re.LOCALE)
    for name in ["query", "product_title", "product_description"]:
        l = line[name]
        # tokenize 分词 只能识别英文，中文无法识别
        tokens = [x.lower() for x in token_pattern.findall(l)]
        # stem 转成词根（转成unicode）
        tokens = stem_tokens(tokens, english_stemmer)
        if exclude_stopword:
            tokens = [x for x in tokens if x not in stopwords]
        tags = pos_tag(tokens)
        tags_list = [t for w, t in tags]
        tags_str = " ".join(tags_list)
        # print tags_str
        line[name] = tags_str
    return line


############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        # 把构造参数传递给父类构造函数
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


# 默认参数
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3


def getTFV(token_pattern=token_pattern,
           norm=tfidf__norm,
           max_df=tfidf__max_df,
           min_df=tfidf__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    """
    tfidf向量
    :param token_pattern:
    :param norm:
    :param max_df:
    :param min_df: 文档中一个词语频率少于该值则被忽略
    :param ngram_range:(1,3)元语法模型
    :param vocabulary:
    :param stop_words:
    :return:
    """
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words=stop_words, norm=norm, vocabulary=vocabulary)
    return tfv


#########
## BOW ##
#########
class StemmedCountVectorizer(CountVectorizer):
    """
    BOW向量
    """

    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


bow__max_df = 0.75
bow__min_df = 3


def getBOW(token_pattern=token_pattern,
           max_df=bow__max_df,
           min_df=bow__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    """

    :param token_pattern:
    :param max_df:
    :param min_df:
    :param ngram_range:
    :param vocabulary:
    :param stop_words:
    :return:
    """
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words=stop_words, vocabulary=vocabulary)
    return bow


################
## Text Clean ##
################
## synonym replacer
replacer = CsvWordReplacer('%s/synonyms.csv' % config.data_folder)
# replacer = CsvWordReplacer('../../Data/synonyms.csv')
# other replace dict
## such dict is found by exploring the training data
replace_dict = {
    "nutri system": "nutrisystem",
    "soda stream": "sodastream",
    "playstation's": "ps",
    "playstations": "ps",
    "playstation": "ps",
    "(ps 2)": "ps2",
    "(ps 3)": "ps3",
    "(ps 4)": "ps4",
    "ps 2": "ps2",
    "ps 3": "ps3",
    "ps 4": "ps4",
    "coffeemaker": "coffee maker",
    "k-cups": "k cup",
    "k-cup": "k cup",
    "4-ounce": "4 ounce",
    "8-ounce": "8 ounce",
    "12-ounce": "12 ounce",
    "ounce": "oz",
    "button-down": "button down",
    "doctor who": "dr who",
    "2-drawer": "2 drawer",
    "3-drawer": "3 drawer",
    "in-drawer": "in drawer",
    "hardisk": "hard drive",
    "hard disk": "hard drive",
    "harley-davidson": "harley davidson",
    "harleydavidson": "harley davidson",
    "e-reader": "ereader",
    "levi strauss": "levi",
    "levis": "levi",
    "mac book": "macbook",
    "micro-usb": "micro usb",
    "screen protector for samsung": "screen protector samsung",
    "video games": "videogames",
    "game pad": "gamepad",
    "western digital": "wd",
    "eau de toilette": "perfume",
}


def clean_text(line, drop_html_flag=False):
    """
    使用正则表达式替换 re.sub
    替换 d gb d g dg -->dgb
    替换 d tb      -->dtb
    替换 replace_dict
    替换 synonyms.csv 同义词
    :param line:
    :param drop_html_flag:
    :return:
    """
    names = ["query", "product_title", "product_description"]
    for name in names:
        l = line[name]
        if drop_html_flag:
            l = drop_html(l)
        l = l.lower()
        # replace gb
        for vol in [16, 32, 64, 128, 500]:
            l = re.sub("%d gb" % vol, "%dgb" % vol, l)
            l = re.sub("%d g" % vol, "%dgb" % vol, l)
            l = re.sub("%dg " % vol, "%dgb " % vol, l)
        # replace tb
        for vol in [2]:
            l = re.sub("%d tb" % vol, "%dtb" % vol, l)
        # replace other words
        for k, v in replace_dict.items():
            l = re.sub(k, v, l)
        l = l.split(" ")
        # replace synonyms
        l = replacer.replace(l)
        l = " ".join(l)
        # 覆盖原来的列
        line[name] = l
    return line


###################
## Drop html tag ##
###################
def drop_html(html):
    """
    删除html标签，保留换行符和制表符  \t\n
    :param html:
    :return:
    """
    return BeautifulSoup(html).get_text(separator=" ")


# token_pattern = r'\w{1,}'
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
def preprocess_data(line, token_pattern=token_pattern, exclude_stopword=config.cooccurrence_word_exclude_stopword, encode_digit=False):
    """
    Pre-process data 预处理每一行，返回处理完后的词根数组
    1.分词
    2.词根
    3.去除停用词
    :param line:
    :param token_pattern:
    :param exclude_stopword:
    :param encode_digit:
    :return:
    """
    # token_pattern = re.compile(token_pattern, flags=re.UNICODE | re.LOCALE)
    token_pattern = re.compile(token_pattern, flags=re.UNICODE)
    # 分词 转小写
    tokens = [x.lower() for x in token_pattern.findall(line)]
    # stem 词根
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    # 去掉停用词
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed
