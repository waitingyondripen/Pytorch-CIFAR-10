'''
@author wyr
@date 2024/05/23
@content 定义LeNet网络
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
    隐藏层网络在torch.nn中
    损失函数在torch.nn中
    激活函数、池化函数在torch.nn中(torch.nn.functional)
    优化器在torch.optim中
'''

#手动实现一个LeNet网络
class LeNet(nn.Module):
    #定义一个初始化函数
    def __init__(self):
        #固定模板，直接套用
        super(LeNet, self).__init__()
        #定义Lenet网络(池化层和激活函数均放在forward，因为其中没有可学习的参数，当然也可以放在__init__中)
        #卷积层的可学习参数有卷积和的权值和各通道的偏置量
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #全连接层的可学习参数有权值和偏置量
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    #定义前向传播函数
    def forward(self, X):
        #传递过程：卷积->激活->池化
        output = self.conv1(X) #卷积
        output = F.relu(output) #Relu激活函数
        output = F.max_pool2d(output, (2, 2)) #池化
        #合写形式
        output = F.max_pool2d(F.relu(self.conv2(output)), 2) 
        #全连接层部分
        #首先需要将tensor转成一维的形式
        output = output.reshape(output.size()[0], -1) #batch_size的每个样本都转成一维的形式
        output = F.relu(self.fc1(output)) #第一层全连接层处理，并使用激活函数relu
        output = F.relu(self.fc2(output)) #第二层全连接层处理，并使用激活函数relu
        output = self.fc3(output)  #最终的输出层处理，不需要激活函数
        '''
        隐藏层激活函数：
            一般选择Relu函数
        '''
        '''
        输出层激活函数的选择：
            Binary Classification,最终可以选择sigmoid函数
            Regression Model, 如果可正可负,则选择linear函数
            Regression Model, 如果不可负,则选择Relu函数
            Multiple Classification, 可以选择softmax函数
        '''
        #返回最终结果
        return output