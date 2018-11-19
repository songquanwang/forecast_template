# -*- coding: utf-8 -*-
import  xdrlib ,sys
import xlrd
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


filename='order_excel_back.xls'
def open_excel(file= filename):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception:
        print (str(Exception))

gap=30
# 根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引
def excel_table_byindex(file=filename, colnameindex=0, by_index=0):
    data = open_excel(file)
    table = data.sheets()[by_index]
    nrows = table.nrows  # 行数
    area_dict={}
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        list = []
        if row:
            serial_str=row[7].split(',')
            #print(serial_str)
            for i in range(len(serial_str)-gap,len(serial_str),1):
                if i<0:
                    continue
                item=[]
                item.append(int(serial_str[i]))
                list.append(item)
        area_dict[row[1]]=list
    return area_dict


def main():
   area_dict = excel_table_byindex()
   dict={}
   ftmp_csv1 = open('order_p1_'+str(gap)+'.csv', 'w')
   ftmp_csv2 = open('order_p2_'+str(gap)+'.csv', 'w')
   ftmp_csv3 = open('order_p3_'+str(gap)+'.csv', 'w')
   ftmp_csv4 = open('order_p4_'+str(gap)+'.csv', 'w')
   ftmp_csv5 = open('order_p5_'+str(gap)+'.csv', 'w')
   ftmp_csv6 = open('order_p6_'+str(gap)+'.csv', 'w')
   ftmp_csv_40 = open('order_'+str(gap)+'.csv', 'w')
   i=0
   for row in area_dict:
       if u'生鲜' in row or '全球购' in row:
           continue
       #row_str=u'华东$杭州市$大件$杭州配送中心'
       ftmp_csv=ftmp_csv6
       if i<10:
           ftmp_csv = ftmp_csv1
       elif i<20:
           ftmp_csv1.close()
           ftmp_csv = ftmp_csv2
       elif i < 30:
           ftmp_csv2.close()
           ftmp_csv = ftmp_csv3
       elif i < 40:
           ftmp_csv3.close()
           ftmp_csv = ftmp_csv4
       elif i < 50:
           ftmp_csv4.close()
           ftmp_csv = ftmp_csv5
       i+=1
       row_str=row
       dataset=area_dict[row_str]
       if len(dataset)<8:
           continue
       # normalize the dataset
       dataset = numpy.array(dataset)
       print('---------------------------')
       print (row_str)
       dataset = dataset.astype('float32')
       scaler = MinMaxScaler(feature_range=(0, 1))
       dataset = scaler.fit_transform(dataset)
       look_back = 1
       train_x, train_y, test_x, test_y=serial2train(dataset)
       print(train_x)
       print(train_y)
       print(test_x)
       print(test_y)
       # train_x = scaler.fit_transform(train_x)
       # train_y = scaler.fit_transform(train_y)
       # test_x = scaler.fit_transform(test_x)
       # test_y = scaler.fit_transform(test_y)
       print('------------------------------------------')
       train_x=numpy.array(train_x)
       train_y=numpy.array(train_y).reshape(1,len(train_y))[0]
       test_x=numpy.array(test_x)
       test_y=numpy.array(test_y).reshape(1,len(train_y))[0]
       print(train_x)
       print(train_y)
       print(test_x)
       print(test_y)
       print('------------------------------------------')
       trainX = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
       testX = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

       # create and fit the LSTM network
       model = Sequential()
       model.add(LSTM(4, input_dim=look_back))
       model.add(Dense(12))
       model.compile(loss='mean_squared_error', optimizer='adam')
       model.fit(trainX, train_y, nb_epoch=100, batch_size=1, verbose=2)

       trainPredict = model.predict(trainX)
       testPredict = model.predict(testX)
       # invert predictions
       trainPredict = scaler.inverse_transform(trainPredict)
       trainY = scaler.inverse_transform([train_y])
       testPredict = scaler.inverse_transform(testPredict)
       testY = scaler.inverse_transform([test_y])
       # calculate root mean squared error
       trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
       print('Train Score: %.2f RMSE' % (trainScore))
       testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
       print('Test Score: %.2f RMSE' % (testScore))
       print(trainY)
       print(trainPredict)
       print(testY)
       print(testPredict)
       print(len(testY[0]))
       print(len(testPredict))
       print('------------------------------------------')
       print(testY[0])
       print('score:'+str(testY[0][len(testY[0])-1])+':'+str(testPredict[len(testPredict)-1]))
       print(str((testPredict[len(testPredict)-1][0]-testY[0][len(testY[0])-1])/testY[0][len(testY[0])-1]))
       score=(testPredict[len(testPredict) - 1][0] - testY[0][len(testY[0]) - 1]) / (testY[0][len(testY[0]) - 1]+1)
       dict[row]=score
       print('------------------------------------------')
       str_w=row_str+','+str(score)+','+str(abs(score))+','+str(testPredict[len(testPredict) - 1][0])+','+str(testY[0][len(testY[0]) - 1])+','+str((testPredict[len(testPredict) - 1][0] - testY[0][len(testY[0]) - 1]))+','+str(len(testPredict))
       ftmp_csv.write(str_w + '\n')
       ftmp_csv_40.write(str_w + '\n')
       print('-----------------------------------------------'+str(i))
   print('================')
   ftmp_csv5.close()
   ftmp_csv6.close()
   ftmp_csv_40.close()

   count=0
   for row in dict:
       if abs(dict[row])<0.5:
           count+=1
       print(dict[row])

   print('================')
   print(len(dict))
   print(count)

#序列转训练数据
def serial2train(list):
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    for i in range(1,len(list)-3,1):
        pass
        t_x=[]
        t_x.append(list[i][0])
        train_x.append(t_x)
        train_y.append(list[i+1][0])
        #测试
        e_x=[]
        e_x.append(list[i+1][0])
        test_x.append(e_x)
        test_y.append(list[i+2][0])
    return train_x,train_y,test_x,test_y


if __name__=="__main__":
    #wb = xlrd.open_workbook(filename)
    # sh = wb.sheet_by_index(0)  # 第一个表 cellName = sh.cell(3,2).value print(cellName)
    # cellName = sh.cell(3, 2).value
    # print(cellName)

    main()