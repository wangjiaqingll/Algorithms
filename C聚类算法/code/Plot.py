# -*- coding: utf-8 -*-
'''
@Time    : 2020.11.21
@Author  : Jiaqingwang
@Email   : wangjiaqing@foxmail.com
'''
import numpy as np
from ColoSelect import ColorSelect
import matplotlib.pyplot as plt


class Plot:
    # 画出散点图
    @staticmethod
    def plot_scatters(result, centers):
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        #ax = plt.subplot(projection='3d',)
        ax.set_xlabel("Height")
        ax.set_ylabel("Weight")
        ax.set_zlabel("50m")
        for i in range(result.__len__()):
            nodes = np.array(result[i])
            center = centers[i]
            nodex = nodes[:, 0]
            nodey = nodes[:, 1]
            nodez = nodes[:, 2]
            centerx = center[0]
            centery = center[1]
            centerz = center[2]
            ax.scatter(nodex,
                       nodey,
                       nodez,
                       c=ColorSelect.nodes_color_select(i),
                       marker="*")
            ax.scatter(centerx,
                       centery,
                       centerz,
                       c=ColorSelect.centers_color_select(i),
                       marker="o")
        plt.title("C-means Algorithm")
        plt.show()
