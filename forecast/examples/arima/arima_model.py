# coding=utf-8
__author__ = 'songquanwang'

import numpy as np
import pandas as pd
import pywt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import matplotlib.pylab as plt

beginDate = '20140101'
end_Date = '20141031'
data_index = np.arange(100, 200)
data_date = np.arange(1, 101)
index_list = data_index[:-10]
date_list1 = data_date[:-10]
index_for_predict = data_index[-10:]
date_list2 = data_date[-10:]
# 分解
A2, D2, D1 = pywt.wavedec(index_list, 'db4', mode='sym', level=2)
coeff = [A2, D2, D1]

# 对各层系数简历Arima模型并重构
# AIC准则,求解模型阶数p,q
order_A2 = sm.tsa.arma_order_select_ic(A2, ic='aic')['aic_min_order']
order_D2 = sm.tsa.arma_order_select_ic(D2, ic='aic')['aic_min_order']
order_D1 = sm.tsa.arma_order_select_ic(D1, ic='aic')['aic_min_order']
print(order_A2, order_D2, order_D1)
# 对每层小波系数构建ARIMA模型
model_A2 = ARIMA(A2, order=order_A2)
model_D2 = ARIMA(D2, order=order_D2)
model_D1 = ARIMA(D1, order=order_D1)

results_A2 = model_A2.fit()
results_D2 = model_D2.fit()
results_D1 = model_D1.fit()

# 画出每层拟合曲线
plt.figure(figsize=(10, 15))
plt.subplot(3, 1, 1)
plt.plot(A2, 'blue')
plt.plot(results_A2.fittedvalues, 'red')
plt.title('model_A2')

plt.title('model_A2')
plt.subplot(3, 1, 2)
plt.plot(D2, 'blue')
plt.plot(results_D2.fittedvalues, 'red')
plt.title('model_D2')

plt.subplot(3, 1, 3)
plt.plot(D1, 'blue')
plt.plot(results_D1.fittedvalues, 'red')
plt.title('model_D1')

# 预测最后10个数据
# 1.计算每个小波系数ARIMA模型需要预测多少步,方法就是查看所有数据小波分解后的系数个数,并求出差值,具体如下:
A2_all, D2_all, D1_all = pywt.wavedec(np.array(data_index), 'db4', mode='sym', level=2)
# 求出差值,则delta序列对应的为每层小波系数ARIMA模型需要预测的步数
delta = [len(A2_all) - len(A2), len(D2_all) - len(D2), len(D1_all) - len(D1)]

print(delta)
# 预测小波系数 包括in-sample和out-sample的需要预测的小波系数
pA2 = model_A2.predict(params=results_A2.params, start=1, end=len(A2) + delta[0])
pD2 = model_D2.predict(params=results_D2.params, start=1, end=len(D2) + delta[1])
pD1 = model_D1.predict(params=results_D1.params, start=1, end=len(D1) + delta[2])

# 重构
coeff_new = [pA2, pD2, pD1]
denoised_index = pywt.waverec(coeff_new, 'db4')

# 画出重构后的原序列预测图
plt.figure(figsize=(15, 5))
plt.plot(index_list, 'blue')
plt.plot(denoised_index, 'red')

# 三、预测的结果
# 10个预测值
temp_data_wt = {
    'real_value': index_for_predict,
    'pre_value_wt': denoised_index[-10:],
    'err_wt': denoised_index[-10:] - index_for_predict,
    'err_rate_wt/%': (denoised_index[-10:] - index_for_predict) / index_for_predict * 100
}
predict_wt = pd.DataFrame(temp_data_wt, index=date_list2, columns=['real_value', 'pre_value_wt', 'err_wt', 'err_rate_wt/%'])
print(predict_wt)

##################
# pacf计算
X = np.array([2, 4, 15, 20])
#XC = sm.add_constant(X)
Y = np.array([1, 2, 3, 4])
#YC = sm.add_constant(Y)
z = np.array([0, 0, 1, 1])
res1 = sm.OLS(z, X).fit()
RES1 = z - res1.predict(X)
res2 = sm.OLS(z, Y).fit()
RES2 = z - res2.predict(Y)
# 相关系数 0.9695015519208121
np.corrcoef(X, Y)[0,1]
# 偏相关系数 0.9087389347953037
np.corrcoef(RES1, RES2)[0,1]
