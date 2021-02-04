# -*- coding: utf-8 -*-
'''
@Time    : 2020.11.21
@Author  : Jiaqingwang
'''
import numpy as np


class Distance:
    # 计算欧式距离
    @staticmethod
    def compute_distance(node, center):
        '''
        input : 向量node,向量center
        output: 两个向量的欧氏距离
        '''
        # 计算欧氏距离有两种方法，取其中一种即可
        # distace1 = np.sqrt(np.sum(np.square(node-center)))
        # distace2 = np.linalg.norm(node-center)
        return np.sqrt(np.sum(np.square(node - center)))
