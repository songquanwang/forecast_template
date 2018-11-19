__author__ = 'songquanwang'
import xgboost as xgb
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # read in data
dtrain = xgb.DMatrix('./agaricus.txt.train')
dtest = xgb.DMatrix('./agaricus.txt.test')
# specify parameters via map
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)


######################3
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
X_train
linreg.fit(X_train, y_train)

def read_svm_file():
    from sklearn.datasets import load_svmlight_file
    X_train, labels_train = load_svmlight_file('D:/github/jc/forecast/theano/xgb/agaricus.txt.train')
import pandas
pandas.Series.diff

