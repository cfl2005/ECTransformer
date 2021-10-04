#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


def test_dat():
    
    # 创建个transform用来处理图像数据
    transform = transforms.Compose([
        transforms.Scale(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    # 下载数据
    train_dataset = dsets.CIFAR10(root='./data/',
                                   train=True,
                                   transform=transform,#用了之前定义的transform
                                   download=True)

    image, label = train_dataset[0]
    print (image.size())
    print (label)


    # data loader提供了队列和线程
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=100,# 这里定义了batch_size
                                   shuffle=True,
                                   num_workers=2)


    # 迭代开始，然后，队列和线程跟着也开始
    data_iter = iter(train_loader)

    # mini-batch 图像 和 标签
    images, labels = next(data_iter)

    for images, labels in train_loader:
        # 这里是训练代码
        pass
if __name__ == '__main__':
    pass

