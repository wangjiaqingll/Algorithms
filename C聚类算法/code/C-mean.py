# -*- coding: utf-8 -*-
'''
@Time    : 2020.11.21
@Author  : Jiaqingwang
'''
import numpy as np
from Plot import Plot
from LoadData import LoadFile
from Classify import ClassifyHandls
# 加载数据集
data, labels = LoadFile.load_csv(r'C聚类算法/dataset/people.csv')

# 随机选择2个样本作为初始聚类中心
class_number = 2
centers = []  # 聚类中心
for i in range(class_number):
    index = np.random.randint(1, len(data), size=1)
    centers.append(data[index])

result = ClassifyHandls.classify(data, centers)
# for i in range(len(result)):
#     print("="*20,"第 %d 类,共 %d 个个体"%(i+1,len(result[i])),"="*20)
#     result[i] = np.array(result[i])
#     print(result[i])
# 求第一类的中心
end_flag = 1
while end_flag:
    new_centers = []
    for i in range(result.__len__()):
        new_centers.append(ClassifyHandls.gen_new_center(result[i]))
    if (np.array(new_centers) == np.array(centers)).all():
        end_flag = 0
    else:
        centers = new_centers.copy()
    result = ClassifyHandls.classify(data, centers)
# for i in range(len(result)):
#     print("="*20,"第 %d 类,共 %d 个个体"%(i+1,len(result[i])),"="*20)
#     result[i] = np.array(result[i])
#     print(result[i])

Plot.plot_scatters(result, centers)
