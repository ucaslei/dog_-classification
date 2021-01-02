import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #指定设备
    print("using {} device.".format(device))

    data_transform = { #数据预处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),# key 为trian 返回这些方法 随机裁剪 224*224
                                     transforms.RandomHorizontalFlip(),#随机反转
                                     transforms.ToTensor(),#转成
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),#标准化处理
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "dog_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])#数据预处理
    train_num = len(train_dataset) #个数

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx #获取名称所对应索引
    cla_dict = dict((val, key) for key, val in flower_list.items()) #遍历 key value 对调
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:#生成json 便于打开
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw) #加载

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True) #类别5

    net.to(device) #网络设备
    loss_function = nn.CrossEntropyLoss() #损失函数
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002) #adam优化器 对象是网络中可训练参数 学习率 自己调参

    save_path = './AlexNet.pth' #保存模型路径
    best_acc = 0.0
    for epoch in range(10):#训练
        # train
        net.train() #管理神经元失活
        running_loss = 0.0 #统计平均损失
        t1 = time.perf_counter() #训练时间
        for step, data in enumerate(train_loader, start=0): #遍历数据集
            images, labels = data #分为图像 标签
            optimizer.zero_grad() #清空梯度信息
            outputs = net(images.to(device)) #正向传播 指定设备
            loss = loss_function(outputs, labels.to(device)) #损失
            loss.backward() #反向传播
            optimizer.step() #更新结点参数

            # print statistics
            running_loss += loss.item() #损失累加
            # print train process
            rate = (step + 1) / len(train_loader) #打印训练进度
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter()-t1)

        # validate
        net.eval() #关闭失活
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1] #最大就是类别
                acc += (predict_y == val_labels.to(device)).sum().item() #预测与真实对比 累加
            val_accurate = acc / val_num #准确率
            if val_accurate > best_acc: #如果准确率大于历史最优
                best_acc = val_accurate #更新
                torch.save(net.state_dict(), save_path) #保存权重
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' % #打印信息
                  (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training')


if __name__ == '__main__':
    main()
