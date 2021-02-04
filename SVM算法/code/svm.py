# -*- encoding: utf-8 -*-
'''
@File    :   svm.py
@Time    :   2021/02/04 08:57:33
@Author  :   Wang Jiaqing
@Contact :   wangjiaqingll@foxmail.com
'''

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix  #计算roc和auc
# from sklearn.model_selection import GridSearchCV
''' 数据读入 '''


# 归一化函数
def maxminnorm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


def svmfun():
    # 数据处理，读入数据
    data = pd.read_csv('SVM算法/dataset/people.csv', header=None, sep=',')
    data = np.array(data)
    np.random.shuffle(data)
    labels = data[..., 0]
    dataset = data[..., 1:7]
    dataset = maxminnorm(dataset)
    x_train, x_test, y_train, y_test = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.2)
    # 做4折交叉验证
    KF = KFold(n_splits=4)
    i = 0
    for train_index, test_index in KF.split(dataset):
        i += 1
        print("=" * 25, "The %d train:" % (i), "=" * 25)
        x_train, x_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # 核函数--参数利用网格搜索法大致确定:
        # linear:gamma=0.1,C=10
        # rbf:C=10,gamma=0.1
        # sigmoid:C=10,gamma=0.1
        # poly:C=10,gamma=10
        clf = SVC(C=10,
                  kernel='linear',
                  degree=3,
                  gamma=0.1,
                  coef0=0.0,
                  shrinking=True,
                  probability=False,
                  tol=1e-3,
                  cache_size=200,
                  class_weight=None,
                  verbose=False,
                  max_iter=-1,
                  decision_function_shape='ovr',
                  random_state=None)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        print('AUC')
        print(roc_auc_score(y_test, y_predict))
        print('ACC')
        print(accuracy_score(y_test, y_predict))
        fpr, tpr, threshold = roc_curve(y_test, y_predict)  #计算真正率和假正率
        roc_auc = auc(fpr, tpr)  #计算auc的值
        matrix = confusion_matrix(y_test, y_predict)  # 计算混淆矩阵
        #print("matrix:")
        TP = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[1][1]
        SE = TP / (TP + FN)
        SP = TN / (TN + FP)
        print("SE:", SE)
        print("SP:", SP)
        #plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=lw,
                 label='ROC curve (area = %0.2f)' %
                 roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Linear-SVM')
        plt.legend(loc="lower right")
        plt.show()


svmfun()
