# -*- coding: utf-8 -*-
'''
@Time    : 2020.11.21
@Author  : Jiaqingwang
'''


class ColorSelect:
    @staticmethod
    def nodes_color_select(index):
        color_dict = ['r', 'g', 'c', 'm']
        return color_dict[index]

    @staticmethod
    def centers_color_select(index):
        color_dict = ['k', 'b', 'y', 'w']
        return color_dict[index]
