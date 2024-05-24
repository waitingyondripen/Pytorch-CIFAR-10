'''
@author wyr
@date 2024/05/23
@content 预加载ResNet模型,训练与测试过程的全实现
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models #用于加载ResNet50预训练模型
import matplotlib.pyplot as plt #用于绘制图像
from Dataset import Data
from tqdm import tqdm

#加载数据集和测试集
trainloader, testloader = Data()

#决定训练器
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#加载神经网络
resnet50 = models.resnet50(weights = 'IMAGENET1K_V1')

#print(resnet50)  #通过该命令可以查看ResNet50的网络结构

#接下来需要冻结网络模型并修改全连接层以适配图像分类任务
num_features = resnet50.fc.in_features  #获取最后全连接层的输入以便于后续更改网络
for params in resnet50.parameters():
    params.requires_grad_(False) #将各种训练参数的梯度训练设置为False，这些参数将不会被更新
#根据实际需求修改全连接层
resnet50.fc = nn.Linear(num_features, 10) #输出为10,修改该层后默认该层的requires_grad = true
resnet50.fc = nn.Sequential(nn.Linear(num_features, 10),
                            nn.LogSoftmax(dim=1)
                            )

#将网络传输给GPU
resnet50.to(device)


#定义标签的标识
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#定义损失函数(交叉熵损失函数，常用于分类问题)
criterion = nn.CrossEntropyLoss()

#定义优化器(使用SGD优化器)
#动量（momentum）参数，较高的动量可以减少参数更新的方差，加速收敛
optimization = optim.SGD(resnet50.parameters(), lr=0.00015, momentum=0.9)

#--------------------------------Train Part--------------------------------#

#创建训练轮次
EPOCHS = 50
#创建数据存储部分
history = {'train_loss':[], 'train_accuracy':[]}
#训练部分核心代码
for epoch in range(1, EPOCHS + 1):
    #设置进度条显示
    process_bar = tqdm(trainloader, unit='step') #unit='step' 参数指定了进度条的单位为步骤，即每次迭代一个批次数据为一个步骤。
    #设置神经网络模式
    resnet50.train() #在训练模式和非训练模式下，有些网络层在训练和推理时具有不同的行为，例如 Dropout 和 Batch Normalization。
    #进入every batch的循环
    for step, (data, label) in enumerate(process_bar):
        #将数据转到合适的设备中
        #Please note that just calling tensor.to(device) returns a new copy of my_tensor on GPU instead of rewriting my_tensor. 
        #You need to assign it to a new tensor and use that tensor on the GPU.
        data = data.to(device)
        label = label.to(device)

        #清空梯度，开始训练
        resnet50.zero_grad()
        #获得神经网络输出
        output = resnet50(data)
        #计算神经网络和标签之间的差异
        loss = criterion(output, label)
        #根据loss进行反向传播
        loss.backward()
        #优化器完成参数优化工作
        optimization.step()

        #计算输出结果的预测值
        predictions = torch.argmax(output, dim=1)
        #判断预测正确的数量
        correct_sum = torch.sum(predictions == label)
        #计算正确率
        accuracy = correct_sum / label.size()[0]

        #将相关信息存储到数组中(每350步存储一次数据)
        if(step % 350 == 349):
            history['train_accuracy'].append(accuracy.item())
            history['train_loss'].append(loss.item())
        
        #进度条输出
        process_bar.set_description("[ %d / %d ], Loss:%.4f, Acc:%.4f"%(epoch, EPOCHS, loss.item(), accuracy.item()))
    
    #每个batch结束后，关闭process_bar
    process_bar.close()


#--------------------------------Test Part--------------------------------#     
correct_sum = 0
total_sum = 0
resnet50.eval() #开启测试模式，此时Dropout以及Batch Normalization不会正常发挥作用
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        output = resnet50(images)
        predictions = torch.argmax(output, dim=1)
        correct_sum += torch.sum(predictions==labels)
        total_sum += images.size()[0]
test_accuracy = correct_sum / total_sum
print("%d张图像的准确率为:%.4f"%(total_sum, test_accuracy.item()))

#--------------------------------save  model--------------------------------# 
torch.save(resnet50, "./ResNet50_Verified.pth") #保留网络的全部信息，包括结构和参数

# torch.save(resnet50.state_dict(), "./ResNet_Verified_state_dict.pth") #只保留网络参数，不保留网络结构

#--------------------------------draw the picture--------------------------------# 
#绘制训练损失变化曲线
plt.plot(history['train_loss'], label = 'Train_Loss')
plt.legend('best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./Train_Loss.png', dpi = 300)
plt.close()

#绘制训练准确率变化曲线
plt.plot(history['train_accuracy'], label = 'Train_Accuracy')
plt.legend('best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('./Train_Accuracy.png', dpi = 300)
plt.close()