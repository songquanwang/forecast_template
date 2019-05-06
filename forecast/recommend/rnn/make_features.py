import pandas as pd
import numpy as np
import datetime as dt
from nnforecast.data_fountain.rnn.feeder import VarFeeder
import numba
from typing import Tuple, Dict, List
import nnforecast.data_fountain.rnn.conf as conf
from sklearn import preprocessing


# category_feature_list = ['cat_level1_id', 'cat_level2_id', 'cat_level3_id', 'brand_id', 'goods_season', 'month', 'day_of_year', 'day_of_month', 'day_of_week']


def get_raw_df(raw_data_path) -> pd.DataFrame:
    """
    Gets source data from start to end date. Any date can be None
    """
    dtypes = {'cat_level1_id': np.float32, 'cat_level2_id': np.float32, 'cat_level3_id': np.float32, 'brand_id': np.float32, 'goods_season': np.float32, 'len': np.float32,
              'start_index': np.int32, 'end_index': np.int32, 'true_len': np.int32,
              }
    df_raw = pd.read_csv(raw_data_path, sep=',', encoding='gbk', dtype=dtypes)
    # 长度>21
    df = df_raw[df_raw['true_len'] >= conf.true_len_limit]
    df['goods_num_list'] = df['goods_num_list'].apply(lambda x: eval(x))

    # 平滑销量
    def smooth_sales(x):
        s = np.std(x)
        m = np.mean(x)
        sx = [x1 if x1 <= m + 2 * s else m for x1 in x]
        return sx

    df['goods_num_list'] = df['goods_num_list'].apply(lambda x: smooth_sales(x))
    df['goods_price_list'] = df['goods_price_list'].apply(lambda x: eval(x))
    df['sales_orginal_shop_price_list'] = df['sales_orginal_shop_price_list'].apply(lambda x: eval(x))
    df = df.set_index('sku_id')
    return df


def get_date_columns_df(raw_df, start_date, data_len, column_name) -> pd.DataFrame:
    """
    日期转成columns
    u'cat_level1_id', u'cat_level2_id', u'cat_level3_id',  u'brand_id',
       u'goods_id', u'sku_id', u'goods_season',
    :param raw_df:
    :param start_date:
    :param len:
    :return:
    """
    columns = [(pd.to_datetime(start_date) + pd.Timedelta(days=da)).strftime('%Y-%m-%d') for da in range(0, data_len * 7, 7)]
    sales_df = pd.DataFrame(columns=columns)
    sales_df[columns] = raw_df.apply(lambda x: pd.Series(x[column_name]), axis=1)
    return sales_df


def get_ts_df(raw_df, start, end) -> pd.DataFrame:
    """
    只获取时间序列df
    :param start:
    :param end:
    :return:
    """
    if start and end:
        return raw_df.loc[:, start:end]
    elif end:
        return raw_df.loc[:, :end]
    else:
        return raw_df


@numba.jit(nopython=True)
def find_start_end(data: np.ndarray):
    """
    Calculates start and end of real traffic data. Start is an index of first non-zero, non-NaN value,
     end is index of last non-zero, non-NaN value
    :param data: Time series, shape [n_pages, n_days]
    :return: 两个长度为(行数)的数组记录每行的起始、结束坐标
    """
    n_pages = data.shape[0]
    n_days = data.shape[1]
    # 保存每行起始坐标
    start_idx = np.full(n_pages, -1, dtype=np.int32)
    end_idx = np.full(n_pages, -1, dtype=np.int32)
    for page in range(n_pages):
        # scan from start to the end
        for day in range(n_days):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                start_idx[page] = day
                break
        # reverse scan, from end to start
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                end_idx[page] = day
                break
    return start_idx, end_idx


def process_data(ts_df, valid_threshold) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    过滤掉很短的时间序列；log1p变换 空值矩阵  每一行起始坐标 结束坐标
    :param start: start date of effective time interval, can be None to start from beginning
    :param end: end date of effective time interval, can be None to return all data
    :param valid_threshold: minimal ratio of series real length to entire (end-start) interval. Series dropped if
    ratio is less than threshold
    :return: tuple(log1p(series), nans, series start, series end)
    """
    # 寻找每一行起始、结束坐标
    starts, ends = find_start_end(ts_df.values)
    # 生成一个数组 过滤实际销量很短的行
    page_mask = (ends - starts) / ts_df.shape[1] < valid_threshold
    print("Masked %d pages from %d" % (page_mask.sum(), len(ts_df)))
    inv_mask = ~page_mask
    # 过滤掉mask中的数据
    df = ts_df[inv_mask]
    nans = pd.isnull(df)
    # log1p变换 空值矩阵  每一行起始坐标 结束坐标
    return df.fillna(0).astype(np.float32), nans, starts[inv_mask], ends[inv_mask]


@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


@numba.jit(nopython=True)
def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_periods = data.shape[1]
    max_end = n_periods - backoffset - 1
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i] + 1
        support[i] = real_len / lag
        if support[i] > threshold:
            series = series[starts[i]:end + 1]
            cm1 = single_autocorr(series, lag)
            c = single_autocorr(series, lag - 1)
            cs1 = single_autocorr(series, lag + 1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c + 0.25 * cm1 + 0.25 * cs1
        else:
            corr[i] = np.NaN
    return corr  # , support


def lag_week_indexes(begin, end, lag_list) -> List[pd.Series]:
    """
    Calculates indexes for 3, 6 months backward lag for the given date range
    :param begin: start of date range
    :param end: end of date range
    :return: List of 4 Series, one for each lag. For each Series, index is date in range(begin, end), value is an index
     of target (lagged) date in a same Series. If target date is out of (begin,end) range, index is -1
    """
    dr = pd.date_range(begin, end, freq='7D')
    # key is date, value is day index
    base_index = pd.Series(np.arange(0, len(dr)), index=dr)

    def lag(offset):
        dates = dr - offset
        return pd.Series(data=base_index.loc[dates].fillna(-1).astype(np.int16).values, index=dr)

    return [lag(pd.DateOffset(weeks=m)) for m in lag_list]


def encode_row_features(df) -> Dict[str, pd.DataFrame]:
    """
    Applies one-hot encoding to page features and normalises result
    :param df: page features DataFrame (one column per feature)
    :return: dictionary feature_name:encoded_values. Encoded values is [n_pages,n_values] array
    """

    def encode(column) -> pd.DataFrame:
        one_hot = pd.get_dummies(df[column], drop_first=False)
        # noinspection PyUnresolvedReferences
        return (one_hot - one_hot.mean()) / one_hot.std()

    return {str(column): encode(column) for column in df}


def normalize(values: np.ndarray):
    return (values - values.mean()) / np.std(values)


def get_max_corr_gap(process_sales_ts_df, starts, ends):
    """
    计算最大的相关系数 gap
    :param process_sales_ts_df:
    :param starts:
    :param ends:
    :return:
    """
    r = []
    for i in range(1, 20):
        autocorr = batch_autocorr(process_sales_ts_df.values, i, starts, ends, 1.5, 0)
        r.append(np.mean(np.abs(autocorr[~np.isnan(autocorr)])))
    max_corr, max_gap = np.max(r), np.argmax(r)
    print(f'max corr is :{max_corr} max gap is :{max_gap}')
    print(r)
    return max_corr, max_gap


def onhot_time_feature(raw_month, raw_day_of_year, raw_day_of_month):
    """
    one hot time features
    :param raw_month:
    :param raw_day_of_year:
    :param raw_day_of_month:
    :return:
    """
    # month
    label_enc_month = preprocessing.LabelEncoder()
    label_enc_month.fit(raw_month)
    month_num = len(label_enc_month.classes_)
    month = label_enc_month.transform(raw_month)
    # day of year
    label_enc_day_of_year = preprocessing.LabelEncoder()
    label_enc_day_of_year.fit(raw_day_of_year)
    day_of_year_num = len(label_enc_day_of_year.classes_)
    day_of_year = label_enc_day_of_year.transform(raw_day_of_year)
    # day of month
    label_enc_day_of_month = preprocessing.LabelEncoder()
    label_enc_day_of_month.fit(raw_day_of_month)
    day_of_month_num = len(label_enc_day_of_month.classes_)
    day_of_month = label_enc_day_of_month.transform(raw_day_of_month)
    return month, day_of_year, day_of_month, month_num, day_of_year_num, day_of_month_num


def run(raw_data_path, tensor_save_dir, start, end, data_len, gap_days):
    # 35天=5 周
    predicted_days = conf.predicted_days
    # 有效数据百分比
    valid_threshold = conf.valid_threshold
    # 计算相关系数 右侧去掉天数
    corr_backoffset = conf.corr_backoffset

    # 获取原始数据
    raw_df = get_raw_df(raw_data_path)
    sales_ts_df = get_date_columns_df(raw_df, start, data_len, column_name='goods_num_list')
    price_ts_df = get_date_columns_df(raw_df, start, data_len, column_name='goods_price_list')
    orginal_price_ts_df = get_date_columns_df(raw_df, start, data_len, column_name='sales_orginal_shop_price_list')

    # 过滤掉有效数据长度不够的
    process_sales_ts_df, nans, starts, ends = process_data(sales_ts_df, valid_threshold)

    filtered_df = raw_df.loc[process_sales_ts_df.index]

    # 起始、结束日期 2017-03-07: 2018-03-13
    data_start, data_end = process_sales_ts_df.columns[-data_len], process_sales_ts_df.columns[-1]

    # 预测最后一周起始时间
    predict_end = (pd.to_datetime(data_end) + pd.Timedelta(predicted_days + gap_days, unit='D')).strftime('%Y-%m-%d')
    print(f"start: {data_start}, end:{data_end}, predict_end:{predict_end}")

    # 找对打的相关系数
    # get_max_corr_gap(process_sales_ts_df, starts, ends)
    # 相关系数 1
    raw_autocorr1 = batch_autocorr(process_sales_ts_df.values, 26, starts, ends, 1.5, corr_backoffset)
    unknown_pct1 = np.sum(np.isnan(raw_autocorr1)) / len(raw_autocorr1)
    # 季度相关系数
    raw_autocorr2 = batch_autocorr(process_sales_ts_df.values, 7, starts, ends, 2, corr_backoffset)
    unknown_pct2 = np.sum(np.isnan(raw_autocorr2)) / len(raw_autocorr2)

    print("Percent of undefined autocorr = autocorr1:%.3f, autocorr2:%.3f" % (unknown_pct1, unknown_pct2))

    # Normalise corr 相关系数转成0
    autocorr1 = normalize(np.nan_to_num(raw_autocorr1))  #
    autocorr2 = normalize(np.nan_to_num(raw_autocorr2))  #

    # 时间相关特征
    features_weeks = pd.date_range(start, freq='7D', periods=54 + 11)
    raw_month = features_weeks.month.values
    raw_day_of_year = features_weeks.dayofyear
    raw_day_of_month = features_weeks.day

    # lagged 7 10 17 34 周lag 注意：包含预测期间的时间范围
    lagged_ix = np.stack(lag_week_indexes(data_start, predict_end, [11, 15, 17, 34]), axis=-1)
    # 一列
    sales_popularity = process_sales_ts_df.median(axis=1)
    sales_popularity = (sales_popularity - sales_popularity.mean()) / sales_popularity.std()

    # Put NaNs back 缺失重置np.NaN
    process_sales_ts_df[nans] = np.NaN

    # one_hot
    category_feature_list = ['cat_level1_id', 'cat_level2_id', 'cat_level3_id', 'brand_id', 'goods_season']

    category_num = {}
    for category_feature in category_feature_list:
        print(category_feature)
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(filtered_df[category_feature])
        category_num[category_feature] = len(label_enc.classes_)
        filtered_df.loc[:, category_feature] = label_enc.transform(filtered_df[category_feature])

    # onhot time features
    month, day_of_year, day_of_month, category_num['month'], category_num['day_of_year'], category_num['day_of_month'] = onhot_time_feature(raw_month, raw_day_of_year,
                                                                                                                                            raw_day_of_month)
    # Assemble final output
    tensors = dict(
        # 三个时间序列
        process_sales_ts=process_sales_ts_df,
        price_ts=price_ts_df,
        orginal_price_ts=orginal_price_ts_df,
        # 时间序列主键 sku_id
        process_sales_ts_ix=process_sales_ts_df.index.values,
        # lagged 坐标数组
        lagged_ix=lagged_ix,
        # 品类信息：need embedding
        cat_level1_id=filtered_df['cat_level1_id'],
        cat_level2_id=filtered_df['cat_level2_id'],
        cat_level3_id=filtered_df['cat_level3_id'],
        brand_id=filtered_df['brand_id'],
        goods_season=filtered_df['goods_season'],
        # 日期特征 ：need embedding
        month=month,  # 12
        day_of_year=day_of_year,
        day_of_month=day_of_month,
        # 总体特征
        popularity=sales_popularity,
        autocorr1=autocorr1,
        autocorr2=autocorr2,
        starts=starts,
        ends=ends,
        true_len=filtered_df['true_len']
    )
    plain = dict(
        features_weeks=len(features_weeks),
        data_weeks=len(process_sales_ts_df.columns),
        n_ts=len(process_sales_ts_df),
        data_start=data_start,
        data_end=data_end,
        predict_end=predict_end,
        category_num=category_num

    )
    # pickle_data_path = './data/data_fountain/tensors.pkl'
    # with open(pickle_data_path, 'wb') as f:
    #     pickle.dump(tensors, f, -1)
    # Store data to the disk
    VarFeeder(tensor_save_dir, tensors, plain)


if __name__ == '__main__':
    """
    [
       u'cat_level1_id', u'cat_level2_id', u'cat_level3_id',  u'brand_id',
       u'goods_id', u'sku_id', u'goods_season', u'data_date',
       u'goods_num',u'goods_price', u'sales_orginal_shop_price'
    ]
     cate特征 =  {
                    'cat_level1_id': 50,
                    'cat_level2_id': 357,
                    'cat_level3_id': 1011,
                    'brand_id': 918 ,
                    'goods_season':7 # [0,1,2,3,4,5,6]
                 }  
     时间相关特征 {
                     'month': 12,
                     'day_of_year': 366,
                     'day_of_month': 31
                     # 'day_of_week': 7
                 }
    暂时没有cate特征，太多，需要embedding
    
    """
    # 7周
    gap_days = 42
    # '2018-05-01'
    predict_start_date = conf.predict_start_date
    # '2017-03-07'
    start = conf.raw_start
    # '2018-03-20' 不包含
    end = (dt.datetime.strptime(predict_start_date, '%Y-%m-%d') - dt.timedelta(gap_days)).strftime('%Y-%m-%d')
    raw_data_path = f'{conf.base_dir}/sales_week_10000.csv'
    # 多少周 54周
    data_len = int(((dt.datetime.strptime(predict_start_date, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')).days - gap_days) / 7)
    tensor_save_dir = f'{conf.base_dir}/vars'
    run(raw_data_path, tensor_save_dir, start, end, data_len, gap_days)
