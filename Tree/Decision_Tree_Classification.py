# @Time    : 2017/6/1 11:00
# @Author  : Yzj
# @Site    : 
# @File    : Decision_Tree_Classification.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np

class Node(object):
    def __init__(self, name, is_leaf, label):
        self.name = name
        self.is_leaf = is_leaf
        self.label = label
        self.child = dict()



def createDataset():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    label = ['no surfacion', 'flippers']
def Load_Data(file_path = ''):
    if file_path == '':
        print('自动生成随机数据集，一个二元分类的简单问题')
        dataSet, label = createDataset()
        return dataSet, label
    else:
        return



