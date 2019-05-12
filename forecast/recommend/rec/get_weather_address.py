# http: // api.k780.com /?app = weather.history & weaid = 1 & date = 2015 - 07 - 20 & appkey = 10003 & sign = b59bc3ef6191eb9f747dd4e83c99f2a4 & format = json


"""

    {
        "weaid":"1",
        "week":"星期一",
        "cityno":"beijing",
        "citynm":"北京",
        "cityid":"101010100",
        "uptime":"2015-07-20 00:50:00",
        "temperature":"22℃",
        "humidity":"97%",
        "aqi":"101",
        "weather":"晴",
        "weather_icon":"http://api.k780.com/upload/weather/d/0.gif",
        "wind":"东北风",
        "winp":"1级",
        "temp":"22",
        "weatid":"1",
        "windid":"13",
        "winpid":"201",
        "weather_iconid":"0"
    }
"""
import datetime as dt
import urllib.request
import pandas as pd


def get_weather_json_data(date):
    # 101270101，天气预报中成都的代码。

    url = 'http://api.k780.com/?app=weather.history&weaid=1&date={0}&appkey=10003&sign=b927d6a3ed8d8af7cb89cd21e8ec7f70&format=json'.format(date)
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    return eval(content)['result']


def gen_time_list(start_date_str, end_date_str):
    """
    生成日期数组
    :param start_date_str:
    :param end_date_str:
    :return:
    """
    start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d')
    date_len = (end_date - start_date).days
    return [(start_date + dt.timedelta(da)).strftime('%Y-%m-%d') for da in range(date_len)]


def get_baidu_location(lnt, lat):
    """
    获取百度地图地址
    116.32,39.96
    116.33,39.79
    :param lat:
    :param lnt:
    :return:
    """
    ak = 'hPOW2xN1TzUTKQUqaLFiehUV0g2ypc36'
    url = "http://api.map.baidu.com/geocoder/v2/?location={},{}&output=json&pois=1&latest_admin=1&ak={}".format(lat, lnt, ak)
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    return eval(content)['result']


def get_gaode_location(lnt, lat):
    """
    获取高德地图地址
    116.32,39.96
    116.33,39.79
    :param lat:
    :param lnt:
    :return:
    """
    ak = 'e076d57e66eaad3e9cfd8bce93d6def5'
    url = "https://restapi.amap.com/v3/geocode/regeo?output=xml&location={},{}&key={}&radius=100&extensions=all".format(lnt, lat, ak)
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    return eval(content)


def get_weather():
    """
    获取三个月天气
    :return:
    """
    date_list = gen_time_list(start_date_str='2018-09-01', end_date_str='2018-09-05')
    all_data = []
    for date in date_list:
        data = get_weather_json_data(date)
        all_data.extend(data)
    weather_df = pd.DataFrame(all_data)
    weather_df.to_csv('weather_20180901-2018-12-31.csv', index=False)


def get_address():
    """
    获取所有地址
    :return:
    """
    lntlat_df = pd.read_csv('./lnt_lat.csv')
    add_list = lntlat_df['lntlat'].values
    data = []
    for add in add_list:
        add_result = get_baidu_location(add.split(',')[0], add.split(',')[1])
        add_dict = add_result['addressComponent']
        add_dict['lntlat'] = add
        data.append(add_dict)
    add_df = pd.DataFrame(data)
    add_df.to_csv('lntlat_address.csv', index=False)


if __name__ == '__main__':
    get_address()
