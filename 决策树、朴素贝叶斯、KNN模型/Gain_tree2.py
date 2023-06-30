import math
import decisionTreePlot as dtPlot
import numpy as np

def createDataSet():
    """
    :return:返回的是创建好的数据集和标签类型
    """
    dataset=[['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460,1],
             ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑',0.774,0.376,1],
             ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑',0.634,0.264,1],
             ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑',0.608,0.318,1],
             ['浅白','蜷缩','浊响','清晰','凹陷','硬滑',0.556,0.215,1],
             ['青绿','稍蜷','浊响','清晰','稍凹','软粘',0.403,0.237,1],
             ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘',0.481,0.149,1],
             ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑',0.437,0.211,1],

             ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑',0.666,0.091,0],
             ['青绿','硬挺','清脆','清晰','平坦','软粘',0.243,0.267,0],
             ['浅白','硬挺','清脆','模糊','平坦','硬滑',0.245,0.057,0],
             ['浅白','蜷缩','浊响','模糊','平坦','软粘',0.343,0.099,0],
             ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑',0.639,0.161,0],
             ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑',0.657,0.198,0],
             ['乌黑','稍蜷','浊响','清晰','稍凹','软粘',0.360,0.370,0],
             ['浅白','蜷缩','浊响','模糊','平坦','硬滑',0.593,0.042,0],
             ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑',0.719,0.103,0]]
    labels=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率','好瓜']
    return dataset,labels
def calculateShannonEnt(dataset,labels):
    """
    :param dataset:
    :return: 返回香农熵
    """
    # 1 计算除了密度和含糖率之外的香农熵
    length=len(dataset)
    yes=0
    # 1.1 计算根节点的香农熵
    for data in dataset:
        if data[-1]==1:
            yes=yes+1
    p_yes=float(yes/length)
    shannonEnt_root=-(p_yes*math.log(p_yes,2)+(1-p_yes)*math.log((1-p_yes),2))
    shannonEnt={}
    rangeNum=0
    if '密度' in labels:
        rangeNum=1
        if '含糖率' in labels:
            rangeNum+=1
    for row in range(len(labels)-rangeNum-1):
        # 1.2 遍历每一列，计算每一列的香农熵
        # featureCounts是记录每个特征的出现的总数以及对应的好瓜次数
        featureCounts = {}
        for column in range(len(dataset)):
            feature=dataset[column][row]
            if feature not in featureCounts:
                featureCounts[feature] = [0, 0]
            featureCounts[feature][0] +=1
            if dataset[column][-1]==1:
                featureCounts[feature][1] += 1
        shannonEnt[row] = 0.0
        for key,value in featureCounts.items():
            p=value[1]/value[0]
            p0=value[0]/length
            if p!=0 and p!=1:
                shannonEnt0=-float(p0*(p*math.log(p,2)+(1-p)*math.log((1-p),2)))
            else:
                shannonEnt0=0
            shannonEnt[row]+=shannonEnt0
    # 2 计算密度和含糖率的熵
    final = {}
    density={}
    sugarContent={}
    # 2.1 获得相应行的密度以及含糖率
    for column in range(len(dataset)):
        if '密度' in labels:
            density_index = labels.index('密度')
            density[column]=dataset[column][density_index]
        if '含糖率' in labels:
            sugarcontent_index = labels.index('含糖率')
            sugarContent[column]=dataset[column][sugarcontent_index]
    density=sorted(density.items(),key=lambda x:x[1])
    sugarContent=sorted(sugarContent.items(),key=lambda x:x[1])
    # 2.2 计算相邻变量的中间值
    middle_density=[]
    middle_sugarContent=[]
    for num in range(len(density)-1):
        middle_density.append((density[num][1]+density[num+1][1])/2)
        middle_sugarContent.append((sugarContent[num][1] + sugarContent[num + 1][1]) / 2)
    # 2.3 计算相应的信息增益并记录划分点
    gain_point_density=calculateENT(shannonEnt_root,middle_density,density,dataset)
    gain_point_sugarContent = calculateENT(shannonEnt_root, middle_sugarContent, sugarContent,dataset)
    # 排序
    gain_point_density=sorted(gain_point_density.items(),key=lambda x:x[1],reverse=True)
    gain_point_sugarContent=sorted(gain_point_sugarContent.items(),key=lambda x:x[1],reverse=True)
    #  3. 计算熵增益
    middle={}
    for key, value in shannonEnt.items():
        final[labels[key]] = shannonEnt_root - value
    if len(gain_point_density)!=0:
        final['密度'] = gain_point_density[0][1]
        middle['密度']=middle_density[gain_point_density[0][0]]
    if len(gain_point_sugarContent)!=0:
        middle['含糖率'] = middle_sugarContent[gain_point_sugarContent[0][0]]
        final['含糖率'] = gain_point_sugarContent[0][1]
        print(middle_density[gain_point_density[0][0]])
    return final,middle
def calculateENT(shannonEnt_root,middle,data,dataset):
    """
    :param shannonEnt_root:根节点的信息熵
    :param middle: 中位数组成的数组
    :param data: 由行和值组成的字典集合
    :return:返回信息增益
    """
    gain={}
    for num in range(len(middle)):
        # 1，计算左右的个数以及好瓜的个数
        # left,right表示middle划分为两类
        left = 0
        right=0
        # num_yes表示左右两边的是好瓜的个数
        num_yes_left=0
        num_yes_right = 0
        middledata=middle[num]
        for key in range(len(data)):
            if data[key][1] < middledata:
                left += 1
                if dataset[data[key][0]][-1]==1:
                    num_yes_left += 1
            if data[key][1] > middledata:
                right += 1
                if dataset[data[key][0]][-1]==1:
                    num_yes_right += 1
        # 2，计算相应的信息熵
        p_left=num_yes_left/left
        p_right=num_yes_right/right
        ent_left=calculate(p_left)
        ent_right=calculate(p_right)
        # 3,计算信息增益
        gain[num]=shannonEnt_root-(left/len(dataset)*ent_left+right*ent_right/len(dataset))
    return gain

def calculate(p):
    """
    :param p:
    :return:返回计算好的信息熵
    """
    shannonEnt=0
    if p!=0 and p!=1.0:
        shannonEnt = -float((p * math.log(p, 2) + (1 - p) * math.log((1 - p), 2)))
    return shannonEnt
def getNumbersByString(dataset,feature,labels):
    """
    :param feature: 输入的特征，如纹理等
    :return:当选中一个特征作为划分节点的时候，需要知道该特征下会有几个特征值，以及每个特征对应的样本编号
    """
    featureSet={}
    index=labels.index(feature)
    for num in range(len(dataset)):
        featureName=dataset[num][index]
        if featureName not in featureSet:
            featureSet[featureName]=[]
        featureSet[featureName].append(num)
    return featureSet
def getNumbers(dataset,feature,labels,middle):
    """
    :param feature: 输入的特征，仅限密度和含糖率
    :param middle: 密度和含糖率的二分点
    :return:当选中一个特征作为划分节点的时候，需要知道该特征下会有几个特征值，以及每个特征对应的样本编号
    """
    featureSet={}
    index=labels.index(feature)
    for num in range(len(dataset)):
        if '小于' not in featureSet:
            featureSet['小于']=[]
        if '大于' not in featureSet:
            featureSet['大于']=[]
        if dataset[num][index]<middle:
            featureSet['小于'].append(num)
        else:
            featureSet['大于'].append(num)
    return featureSet
def modifyLabels(labels,dataset,labelName):
    """
    :param labels:输入的标签组合
    :param dataset:输入的数据集合
    :param labelName: 需要删除的标签名称
    :return: 返回新的标签组合和数据集合
    """
    # 生成新的标签集合
    num=labels.index(labelName)
    del (dataset[num])
    return dataset
def modifyDataset(dataset,nums):
    """
    这个函数主要是在选择了将某个节点作为根节点（如纹理）划分之后
    其必然有多个属性值（如清晰、稍糊、模糊等），根据将对应的属性值的样本划分到新的数据集中
    :param dataset: 输入进来的dataset
    :param nums: 如选择纹理作为划分，清晰对应的行有：[0, 1, 2, 3, 4, 5, 7, 9, 14]
    :return: 将[0, 1, 2, 3, 4, 5, 7, 9, 14]对应的行整理输出作为新的数据集
    """
    newDataset=[]
    for num in range(len(nums)):
        newDataset.append(dataset[nums[num]])
    return newDataset
def DeleteFeaturesAndLabels(dataset,labels,feature):
    """
    :param dataset:输入进来的总的dataset
    :param labels: 输入进来总的标签集合
    :param feature: 选择分析的标签名，如纹理
    :return: 删除了纹理对应的属性值（如清晰、稍糊、模糊等）之后新的数据集合标签集
    """
    num = labels.index(feature)
    for key,value in dataset.items():
        for num0 in range(len(value)):
            del(value[num0][num])
    # 将labels对应列删除
    del (labels[num])
    return dataset,labels

def createTree(dataset, labels):
    """
    :return:创建分类树
    """
    #类别列表
    classList=[example[-1] for example in dataset]
    # 若全部都是一个类的时候结束，为叶结点
    if classList.count(classList[0])==len(classList):
        if classList[0]==1:
            return labels[-1] #好瓜
        else:
            return '坏瓜'
    # 若只有最后一个样本的时候，为叶结点
    if len(dataset[0])==1:
        if dataset[-1]==1:
            return labels[-1]
        else:
            return '坏瓜'
    #计算信息熵
    final, middle = calculateShannonEnt(dataset, labels)

    final = sorted(final.items(), key=lambda x: x[1], reverse=True)

    feature = final[0][0]

    featureSet = {}
    middleNum=0.0
    if feature=='密度':  #连续属性
        middleNum=middle['密度']
        featureSet=getNumbers(dataset,feature,labels,middleNum)
    elif feature=='含糖率':  #连续属性
        middleNum=middle['含糖率']
        featureSet = getNumbers(dataset, feature, labels, middleNum)
    else:  #通过串
        featureSet= getNumbersByString(dataset,feature,labels)
    print(featureSet)
    #新数据集
    newdataset={}
    myTree = {feature: {}}
    for key,value in featureSet.items():
        newdataset[key]=modifyDataset(dataset,value)
    #删除特征和属性
    newdataset,labels=DeleteFeaturesAndLabels(newdataset,labels,feature)
    for key,value in newdataset.items():
        #递归创建决策树
        myTree[feature][key]=createTree(value,labels)
    return myTree

def majorityCnt(dataset, labels):
    """
    :param dataset:输入的数据集
    :param labels:输入的标签集合
    :return: 返回的是选择分类的标签名
    """
    final,middle = calculateShannonEnt(dataset, labels)
    final = sorted(final.items(), key=lambda x: x[1], reverse=True)
    feature = final[0][0]
    return feature
if __name__ == "__main__":
    dataset, labels = createDataSet()
    myTree = createTree(dataset, labels)
    dataset, labels1 = createDataSet()
    dtPlot.createPlot(myTree)
