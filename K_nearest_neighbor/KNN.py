import numpy as np
import operator
import matplotlib.pyplot as plt


def Knn_Classifier(inX, dataSet, labels, k):
    """
    Knn_Classifier k近邻核心实现代码
    :param inX: 输入待测试实例
    :param dataSet: 输入的训练数据集
    :param labels: 训练数据集对应的标签
    :param k: 分为k类
    :return: 返回对应的类别
    """
    dataSetSize = dataSet.shape[0]  # 获取数据集的
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 计算inx与训练数据集中所有数据的差值
    sqDiffMat = diffMat**2  # 这里采用的是欧氏距离
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = np.argsort(distances)  # 按照欧氏距离的大小进行排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 依次存入欧氏距离前k个最小距离的编号
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 对应一个字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 对字典进行排序
    return sortedClassCount[0][0]  # 最高值对应类别即为inX的类别

def file2matrix(filename):
    f = open(filename)
    arrayOLines = f.readlines()
    numberOfLines = len(arrayOLines)
    numberOfFeature = len(arrayOLines[1].split('\t'))
    returnMat = np.zeros((numberOfLines, numberOfFeature - 1))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromline = line.split('\t')
        returnMat[index, :] = listFromline[0:3]
        classLabelVector.append(listFromline[-1])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    autoNorm 归一化函数，消除量纲不同，不同属性的大小差异过大的影响
    归一化公式 newValue = (oldValue - min)/(max - min)
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 广播操作
    normDataSet = normDataSet/(np.tile(ranges, (m, 1)))  # 广播操作

    return normDataSet, ranges, minVals

def datingClassTest():
    """
    datingClassTest 测试约会数据集datingTestSet2的聚类预测结果
    return: null
    """
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读入数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化数据
    m = normMat.shape[0]  # 获取行数
    numTestVecs = int(m * hoRatio)  # 取出其中的10%作为测试数据
    errorCount = 0.0  # 记录错误率
    for i in range(numTestVecs):
        classifierResult = Knn_Classifier(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)  #利用KNN聚类算法进行预测
        print("the classifier came back with: " +classifierResult + " the real answer is: " + datingLabels[i])
        if(classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))



def createDataSet():
    """
    createDataSet 小数据集生成demo
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0., 0.], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__ == "__main__":
    # group, lables = createDataSet()
    # print(Knn_Classifier([0, 0], group, lables, 3))
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # print(datingDataMat)
    # print(datingLabels[:20])
    # print(autoNorm(datingDataMat)[0])
    # print(autoNorm(datingDataMat)[1])
    # print(autoNorm(datingDataMat)[2])
    datingClassTest()