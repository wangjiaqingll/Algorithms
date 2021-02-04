# -*- coding: utf-8 -*-
'''
@Time    : 2020.11.21
@Author  : Jiaqingwang
'''
import numpy as np
from Distance import Distance


class ClassifyHandls:
    @staticmethod
    def classify(nodes, centers):
        result = []
        for i in range(centers.__len__()):
            result.append([])

        for i in range(nodes.__len__()):
            min_distance = float('inf')
            class_index = 0
            for j in range(centers.__len__()):
                distance = Distance.compute_distance(nodes[i], centers[j])
                if min_distance > distance:
                    min_distance = distance
                    class_index = j
            result[class_index].append(nodes[i])

        return result

    # 计算每一类的中心
    @staticmethod
    def gen_new_center(nodes):
        # 初始类中心选为第一个数据位置
        # 聚类中心是每个特征算术平均值组成的向量
        nodes = np.array(nodes)
        center = np.mean(nodes, axis=0)
        return center
