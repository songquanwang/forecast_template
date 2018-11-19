__author__ = 'songquanwang'

param = {
	'task': 'regression',
	'booster': 'gbtree',
	'objective': 'reg:logistic',
	'eta': 0.1,
	'gamma': 1.1,
	'min_child_weight': 2.0,
	'max_depth': 17,
	'subsample': 0.7,
	'colsample_bytree': 0.8,
	'num_round': 30,
	'seed': 2017,
	"max_evals": 200

}


import xgboost as xgb
import numpy as np
# read in data
d_arr =np.array([[0,0,0,0,1,1,1,1,1],[150,180,130,120,100,120,110,90,80]]).T
label_arr =np.array([1,0,1,0,1,0,1,1,0])



dtrain =xgb.DMatrix(d_arr, label=label_arr)

dtest = xgb.DMatrix(np.array([[0,135],[1,95]]))
# specify parameters via map

bst = xgb.train(param, dtrain, 200)
# make prediction
preds = bst.predict(dtest)

xgb.plot_tree(bst)


#####################################

import matplotlib.pyplot as plt
param = {
	'task': 'regression',
	'booster': 'gblinear',
	'objective': 'reg:linear',
	'eta': 0.1,
	'gamma': 1.1,
	'min_child_weight': 2.0,
	'max_depth': 17,
	'subsample': 0.7,
	'colsample_bytree': 0.8,
	'num_round': 5,
	'seed': 2017,
	"max_evals": 5

}

d_arr =np.array([range(20),range(100,300,10)]).T
label_arr =3*d_arr[:,0]+2*d_arr[:,1]



dtrain =xgb.DMatrix(d_arr, label=label_arr)

dtest = xgb.DMatrix(np.array([[12,13],[20,15]]))
# specify parameters via map

dtest1=xgb.DMatrix([[  0, 100],[  1, 110]])
bst = xgb.train(param, dtrain, 5)
# make prediction
preds = bst.predict(dtest)

xgb.plot_tree(bst)

bst.dump_model('model_liner.txt')


######################3
from sklearn.linear_model import LinearRegression
import random
linreg = LinearRegression()


X_train=np.array([[random.randint(1,100) for i in range(100)],[random.randint(200,300) for i in range(100)]]).T
y_train=3*X_train[:,0]+2*X_train[:,1]

# y_train=d_arr[:,0]+d_arr[:,1]
linreg.fit(X_train, y_train)
X_test =np.array([[12,13],[20,15]])
y_pred = linreg.predict(X_test)


print linreg.intercept_
print linreg.coef_

y_pred = linreg.predict(X_train)

fig, ax = plt.subplots()
ax.scatter(y_train, y_pred)
