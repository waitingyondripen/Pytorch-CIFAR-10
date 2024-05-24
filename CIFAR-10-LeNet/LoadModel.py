'''
@author wyr
@date 2023/05/24
@content 加载预存模型,复现结果,设计两种不同的加载方式(加载全部or只加载网络参数)
'''
import torch
from Dataset import Data
from Network import LeNet

#加载数据集和测试集(这里只用到测试集)
trainloader, testloader = Data()

#设置计算装置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model.pth存储的是整个完整的网络模型
#model_state_dict.pth存储的是网络参数,没有网络结构

#无需自导入网络结构
#直接加载整体网络模型
network = torch.load('model.pth')

#需要首先自载入网络结构
#再导入网络参数
# network = LeNet()
# network.load_state_dict(torch.load('model_state_dict.pth'))

network.to(device)

#--------------------------------Test Part--------------------------------#     
correct_sum = 0
total_sum = 0
#network.eval() #开启测试模式，此时Dropout以及Batch Normalization不会正常发挥作用
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        output = network(images)
        predictions = torch.argmax(output, dim=1)
        correct_sum += torch.sum(predictions==labels)
        total_sum += images.size()[0]
test_accuracy = correct_sum / total_sum
print("%d张图像的准确率为:%.4f"%(total_sum, test_accuracy.item()))