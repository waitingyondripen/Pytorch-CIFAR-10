'''
@author wyr
@date 2024/05/24
@content 加载训练集和测试集(迁移学习版)
'''
import torch
import torch.utils
import torch.utils.data
import torchvision


def Data():
    '''
    No input params;
    Output params : trainloader, testloader
    '''

    #定义对数据的预处理
    transform = torchvision.transforms.Compose([
        #将数据转化为Tensor(张量)
        torchvision.transforms.ToTensor(),
        #对数据进行标准化处理
        #接受三个参数：mean(均值)、std(标准差)、inplace(可选参数，用于指定是否直接在原始图像数据上进行标准化操作)
        #标准化的过程是将每个像素值减去均值，然后除以标准差，从而使得数据分布接近标准正态分布
        #这有助于提高模型的训练速度和稳定性
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    #Dataset的作用在于将数据和标签以各种形式存储
    #DataLoader的作用在于将数据打包，以便于送入GPU中，打包和送入GAU异步操作，提高效率

    #加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='/data/jyc/pytorch-CIFAR10-datasets/', #指定数据存储位置  #注意，在这里需要修改地址到合适的存储位置
        train=True, #指定加载训练集还是测试集。true为训练集，false为测试集
        download=True, #如果数据集尚未下载到 root 指定的路径下，则会在加载时自动下载。
        transform=transform #指定对加载图形进行的转换操作
        #@param target_transform  #在此没有指定，其作用是指定对目标标签进行的转换操作
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, #指定要加载的数据集对象
        batch_size = 32, #指定每个批次的样本数目
        shuffle=True,  #一个布尔值，指定是否在每个epoch时打乱数据顺序
        num_workers=2 #指定用于数据加载的子进程数目。通过多进程加载数据可以加速数据加载过程，特别是对于大规模数据集
        #@param pin_memory  #一个布尔值，指定是否将数据加载到 CUDA 固定内存中，以加速 GPU 加速
        #@drop_last   #一个布尔值，指定是否丢弃最后一个批次中样本数目不足一个批次的数据
    )

    #加载测试集
    testset = torchvision.datasets.CIFAR10(
        root='/data/jyc/pytorch-CIFAR10-datasets/',   #注意，在这里需要修改地址到合适的存储位置
        train=False, 
        download=True, 
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        dataset=testset, 
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    return trainloader, testloader

