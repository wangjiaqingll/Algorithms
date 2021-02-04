# -*- coding: utf-8 -*-
'''
@Time    : 2020.11.21
@Author  : Jiaqingwang
'''
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)


# 文件操作类
class LoadFile:
    # 按列归一化特征,公式(n为某列样本集合)：x' = (x-min(n)) / (max(n)-min(n))
    @staticmethod
    def maxminnorm(array):
        '''
        input : 未归一化的矩阵
        output: 归一化后的矩阵
        '''
        array = np.array(array)
        maxcols = array.max(axis=0)
        mincols = array.min(axis=0)
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        new_array = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            new_array[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] -
                                                            mincols[i])
        return new_array

    # 读取csv文件
    @staticmethod
    def load_csv(filename):
        '''
        input : csv文件路径
        output: 数据集，标签
        '''
        data = pd.read_csv(filename, sep=',')
        data = np.array(data)
        np.random.shuffle(data)  # 打乱数据集
        labels = data[..., 0]  # 选择标签
        dataset = data[..., [1, 2, 4]]  # 选择需要的数据，按列
        #dataset = LoadFile.maxminnorm(dataset) # 归一化
        return dataset, labels
