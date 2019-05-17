from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


###########################聚类
def gen_cluster(cluster_num, X):
    # 将数据聚类成2个
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    return y_pred, kmeans


X = od_df[['lnt', 'lat']].values
fig = plt.figure(figsize=(10, 6))
y_pred, kmeans = gen_cluster(cluster_num=10, X=X)
# 画出簇分配和簇中心
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', s=100, linewidth=2)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#######################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib as mpl

mpl.rcParams['legend.fontsize'] = 10

# 数据

values = plans_df[['distance', 'eta', 'price', 'transport_mode']].sample(frac=0.01, random_state=1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib as mpl

mpl.rcParams['legend.fontsize'] = 10

##################################数据  多维散点图

values = plans_df[['distance', 'eta', 'price', 'transport_mode']]


def make_data(values):
    x = values[:, 0]
    y = values[:, 1]
    z = values[:, 2]
    c = values[:, 3]
    return x, y, z, c


values1 = values[values['transport_mode'] == 1].sample(frac=0.05, random_state=1).values
values2 = values[values['transport_mode'] == 2].sample(frac=0.05, random_state=1).values
values3 = values[values['transport_mode'] == 3].sample(frac=0.05, random_state=1).values
values4 = values[values['transport_mode'] == 4].sample(frac=0.05, random_state=1).values
values5 = values[values['transport_mode'] == 5].sample(frac=0.05, random_state=1).values
values6 = values[values['transport_mode'] == 6].sample(frac=0.05, random_state=1).values
values7 = values[values['transport_mode'] == 7].sample(frac=0.05, random_state=1).values
values8 = values[values['transport_mode'] == 8].sample(frac=0.05, random_state=1).values
values9 = values[values['transport_mode'] == 9].sample(frac=0.05, random_state=1).values
values10 = values[values['transport_mode'] == 10].sample(frac=0.05, random_state=1).values
values11 = values[values['transport_mode'] == 11].sample(frac=0.05, random_state=1).values

fig = plt.figure(figsize=(20, 15))

colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo', 'tan']
# ax = Axes3D(fig)
ax = fig.gca(projection='3d')
#
x1, y1, z1, c1 = make_data(values1)
p1 = ax.scatter(x1, y1, z1, c=colors[0], s=60)

x2, y2, z2, c2 = make_data(values2)
p2 = ax.scatter(x2, y2, z2, c=colors[1], s=60)

x3, y3, z3, c3 = make_data(values3)
p3 = ax.scatter(x3, y3, z3, c=colors[2], s=60)

x4, y4, z4, c4 = make_data(values4)
p4 = ax.scatter(x4, y4, z4, c=colors[3], s=60)

x5, y5, z5, c5 = make_data(values5)
p5 = ax.scatter(x5, y5, z5, c=colors[4], s=60)

x6, y6, z6, c6 = make_data(values6)
p6 = ax.scatter(x6, y6, z6, c=colors[5], s=60)

x7, y7, z7, c7 = make_data(values7)
p7 = ax.scatter(x7, y7, z7, c=colors[6], s=60)

x8, y8, z8, c8 = make_data(values8)
p8 = ax.scatter(x8, y8, z8, c=colors[7], s=60)

x9, y9, z9, c9 = make_data(values9)
p9 = ax.scatter(x9, y9, z9, c=colors[8], s=60)

x10, y10, z10, c10 = make_data(values10)
p10 = ax.scatter(x10, y10, z10, c=colors[9], s=60)

x11, y11, z11, c11 = make_data(values11)
p11 = ax.scatter(x11, y11, z11, c=colors[10], s=60)

# 添加坐标轴
ax.set_xlabel(u'X-distance', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel(u'Y-eta', fontdict={'size': 15, 'color': 'blue'})
ax.set_zlabel(u'Z-price', fontdict={'size': 15, 'color': 'black'})
ax.set_xlim3d(0, 100000)
ax.set_ylim3d(0, 25000)
ax.set_zlim3d(0, 10000)
ax.legend([p1, p2, p3, p5, p6, p7, p8, p9, p10, p11], ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11'], numpoints=1)


############输出 分布对比图
def sort_count(c):
    d = dict(c.transport_mode.value_counts())
    ds = [(k, d[k]) for k in sorted(d.keys())]
    x, y = zip(*ds)
    return x, y


def sort_count(c):
    d = dict(c)
    ds = [(k, d[k]) for k in sorted(d.keys())]
    x, y = zip(*ds)
    return x, y


import pandas as pd

pred_df = pd.read_csv('../submit/gbdt_ext_valid_result_2019-05-16-07-22-32.csv')
plt.figure(figsize=(10, 6))
width = 0.3
c1 = pred_df['click_mode'].value_counts()
c2 = pred_df['recommend_mode'].value_counts()
x1, y1 = sort_count(c1)
x2, y2 = sort_count(c2)
y11 = y1 / sum(y1)
y22 = y2 / sum(y2)

plt.bar(x1 - width / 2, y11, width=0.3)
plt.bar(x2 + width / 2, y22, width=0.3)
