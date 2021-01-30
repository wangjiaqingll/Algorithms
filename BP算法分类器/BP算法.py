#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: JiaqingWang
@time: 2021/01/29
"""
import pandas as pd
from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix
from sklearn.model_selection import KFold


# 归一化函数
def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

# 数据集读取与处理
def loadDataset(filepath):
    """加载数据集，对数据进行预处理，并打乱数据集
        filepath: 数据集文件存放路径
    """
    pydata = pd.read_csv(filepath)
    # 填充缺省值--用各列平均数补全
    pydata['ShoeSize'] = pydata['ShoeSize'].fillna(pydata['ShoeSize'].mean())
    pydata['_50m'] = pydata['_50m'].fillna(pydata['_50m'].mean())
    pydata['Pulmonary'] = pydata['Pulmonary'].fillna(pydata['Pulmonary'].mean())
    data = pydata.dropna()
    data = data.iloc[:,[0,1,2,3,4,5,6]].values
    dataset = data[:,[0,2,3,4,5,6]]
    dataset = np.array(dataset)
    dataset = maxminnorm(dataset)
    # labels = data[..., 0]
    # labels = np.array(labels)
    return dataset

def fun_z(weights, inputs):
    """计算神经元的输入：z = weight * inputs + b
    :param weights: 网络参数（权重矩阵和偏置项）
    :param inputs: 上一层神经元的输出
    :return: 当前层神经元的输入
    """
    bias_term = weights[-1]
    z = 0
    for i in range(len(weights)-1):
        z += weights[i] * inputs[i]
    z += bias_term
    return z


def sigmoid(z):
    """激活函数(Sigmoid)：f(z) = Sigmoid(z)
    :param z: 神经元的输入
    :return: 神经元的输出
    """
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(output):
    """Sigmoid激活函数求导
    :param output: 激活函数的输出值
    :return: 求导计算结果
    """
    return output * (1.0 - output)


def forward_propagate(network, inputs):
    """前向传播计算
    :param network: 神经网络
    :param inputs: 一个样本数据
    :return: 前向传播计算的结果
    """
    for layer in network:     # 循环计算每一层
        new_inputs = []
        for neuron in layer:  # 循环计算每一层的每一个神经元
            z = fun_z(neuron['weights'], inputs)
            neuron['output'] = sigmoid(z)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, actual_label):
    """误差进行反向传播
    :param network: 神经网络
    :param actual_label: 真实的标签值
    :return:
    """
    for i in reversed(range(len(network))):  # 从最后一层开始计算误差
        layer = network[i]
        errors = list()
        if i != len(network)-1:  # 不是输出层
            for j in range(len(layer)):  # 计算每一个神经元的误差
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:  # 输出层
            for j in range(len(layer)):  # 计算每一个神经元的误差
                neuron = layer[j]
                errors.append(actual_label[j] - neuron['output'])
        # 计算误差项 delta
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])


def update_parameters(network, row, l_rate):
    """利用误差更新神经网络的参数（权重矩阵和偏置项）
    :param network: 神经网络
    :param row: 一个样本数据
    :param l_rate: 学习率
    :return:
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:  # 获取上一层网络的输出
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            # 更新权重矩阵
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # 更新偏置项
            neuron['weights'][-1] += l_rate * neuron['delta']


def initialize_network(n_inputs, n_hidden, n_outputs):
    """初始化BP网络（初始化隐藏层和输出层的参数：权重矩阵和偏置项）
    :param n_inputs: 特征列数
    :param n_hidden: 隐藏层神经元个数
    :param n_outputs: 输出层神经元个数，即分类的总类别数
    :return: 初始化后的神经网络
    """
    network = list()
    # 隐藏层
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # 输出层
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def train(train_data, l_rate, epochs, n_hidden):
    """训练神经网络（迭代n_epoch个回合）
    :param train_data: 训练集
    :param l_rate: 学习率
    :param epochs: 迭代的回合数
    :param n_hidden: 隐藏层神经元个数
    :param val_data: 验证集
    :return: 训练好的网络
    """
    # 获取特征列数
    n_inputs = len(train_data[0]) - 1
    # 获取分类的总类别数
    n_outputs = len(set([row[0] for row in train_data]))
    # 初始化网络
    network = initialize_network(n_inputs, n_hidden, n_outputs)

    for epoch in range(epochs):  # 训练epochs个回合
        for row in train_data:
            # 前馈计算
            _ = forward_propagate(network, row)
            # 处理一下类标，用于计算误差
            actual_label = [0 for i in range(n_outputs)]
            row = row.tolist()
            actual_label[int(row[0])] = 1
            # 误差反向传播计算
            backward_propagate_error(network, actual_label)
            # 更新参数
            update_parameters(network, row, l_rate)
    return network


def validation(network, val_data):
    """测试模型在验证集上的效果
    :param network: 神经网络
    :param val_data: 验证集
    :return: 模型在验证集上的准确率
    """
    # 获取预测类标
    predicted_label = []
    for row in val_data:
        prediction = predict(network, row)
        predicted_label.append(prediction)
    # 获取真实类标
    actual_label = [row[0] for row in val_data]
    # 计算SE，SP，ACC指标
    SE, SP, ACC = accuracy_calculation(actual_label, predicted_label)
    print("SE:",SE)
    print("SP:",SP)
    print("ACC:",ACC)
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(actual_label, predicted_label)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    # 假正率为横坐标，真正率为纵坐标做曲线
    lw = 3
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BP Algorithm')
    plt.legend(loc="lower right")
    plt.show()

def accuracy_calculation(actual_label, predicted_label):
    """计算准确率
    :param actual_label: 真实类标
    :param predicted_label: 模型预测的类标
    :return: 准确率（百分制）
    """
    correct_count = 0
    for i in range(len(actual_label)):
        if actual_label[i] == predicted_label[i]:
            correct_count += 1

    # 计算混淆矩阵
    matrix = confusion_matrix(actual_label,predicted_label)
    TP = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[1][1]
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)
    ACC = (TP+TN)/(TP+FP+TN+FN)

    return SE,SP,ACC


def predict(network, row):
    """使用模型对当前输入的数据进行预测
    :param network: 神经网络
    :param row: 一个数据样本
    :return: 预测结果
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


if __name__ == "__main__":
    file_path = 'BP算法分类器/dataset.csv'
    
    l_rate = 0.2        # 学习率
    epochs = 150        # 迭代训练的次数
    n_hidden = 5        # 隐藏层神经元个数

    # 加载数据并划分训练集和验证集
    dataset = loadDataset(file_path)
    #5折交叉划分数据集
    kf = KFold(n_splits=5,shuffle=True,random_state=0)
    # 训练模型
    i = 1
    for train_index, test_index in kf.split(dataset):
        print("*********Train %d**********"%(i))
        train_data = dataset[train_index]
        val_data = dataset[test_index]
        network = train(train_data, l_rate, epochs, n_hidden)
        validation(network, val_data)
        i += 1
    