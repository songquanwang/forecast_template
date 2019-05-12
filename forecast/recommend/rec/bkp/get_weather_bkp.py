#http: // api.k780.com /?app = weather.history & weaid = 1 & date = 2015 - 07 - 20 & appkey = 10003 & sign = b59bc3ef6191eb9f747dd4e83c99f2a4 & format = json


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
import json
import urllib.request
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime


def get_weather_json_data(date):
    # 101270101，天气预报中成都的代码。

    url = 'http://api.k780.com/?app=weather.history&weaid=1&date={0}&appkey=10003&sign=b59bc3ef6191eb9f747dd4e83c99f2a4&format=json'.format(date)
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    return content


def get_forecast_data(content):
    data = content["data"]
    return data["result"]


# 清洗数据。过滤℃
def get_pure_temperature(temp):
    a, b = temp.split()
    return b.strip().strip("℃")


def make_chart(high, low, date_time):
    matplotlib.rc('font', family='SimHei', weight='bold')
    plt.rcParams['axes.unicode_minus'] = False

    x = range(len(date_time))

    plt.plot(x, low, ms=10, marker='*', color='blue', alpha=0.5, label="低温")
    plt.plot(x, high, ms=10, marker='o', color='red', alpha=0.5, label="高温")

    plt.fill_between(x, low, high, facecolor='gray', alpha=0.1)
    plt.title("2019年3月 - 温度变化", fontsize=15)

    plt.xticks(x, date_time, rotation=20)

    plt.xlabel('日期')
    plt.ylabel('温度')
    plt.grid()  # 显示网格
    plt.legend()
    plt.show()


content = json.loads(get_weather_json_data())
data = get_forecast_data(content)

high, low, date_time = [], [], []
for obj in data:
    h = obj["high"]
    high.append(get_pure_temperature(h))

    l = obj["low"]
    low.append(get_pure_temperature(l))

    date_time.append(obj["ymd"])

# 取得高温低温和日期，开始绘折线图。
make_chart(high, low, date_time)
