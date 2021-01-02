import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  #可以将层结构进行打包
            #卷积核大小11，卷积核个数48（数据集小，加快训练速度） 彩色图片 深度是3
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            #激活函数 inplace增加计算量 降低内存使用
            nn.ReLU(inplace=True),
            #池化层
            nn.MaxPool2d(kernel_size=3, stride=2),   # output[48, 27, 27]
            #第二个卷积层 卷积默认stride是1 不用设置 经过上一层得到48 原论文一半
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            #第三个最大采样
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            #128同输出 192是一半
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            #卷积4
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            #卷积5
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            #池化核大小3
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential( #分类器 全连接层
            nn.Dropout(p=0.5), #部分神经元失活  防止过拟合 失活比例0.5
            nn.Linear(128 * 6 * 6, 2048),#展平成1维  结点个数2048
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048), #全连接2
            nn.ReLU(inplace=True), #全连接3 输出层
            nn.Linear(2048, num_classes), #数据集种类 初始化传入
        )
        if init_weights: #初始化权重 初始化为True 则进入
            self._initialize_weights()

    def forward(self, x): #正向传播 x输入进来的变量
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) #展平 从高度宽度展成1维
        x = self.classifier(x) #分类
        return x

    def _initialize_weights(self):
        for m in self.modules(): #遍历模块  何恺明
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) #全连接层 正态分布
                nn.init.constant_(m.bias, 0)
