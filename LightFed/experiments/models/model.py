import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
# from transformers import BertModel


def model_pull(args):


    if args.model_type == 'LR':  
        return LR(args)
    
    elif args.model_type == 'TinyNN':  
        return TinyNN(args)

    elif args.model_type == 'Lenet_5':  
        return Lenet()

    elif args.model_type == 'ResNet_20':  
        return ResNet(BasicBlock, [3, 3, 3], args)

    else:
        raise Exception(f"unkonw model_type: {args.model_type}")



'''凸模型：LR MINST'''
class LR(torch.nn.Module):
    def __init__(self, args):
        super(LR, self).__init__()
        self.class_num = args.data_distributer.class_num
        self.feature_size = np.prod(args.data_distributer.x_shape)
        self.data_set = args.data_set

        self.fc = nn.Linear(in_features=self.feature_size,
                            out_features=self.class_num, bias=True)

    def forward(self, x):
        if self.data_set in ['MNIST', 'CIFAR-10', 'CIFAR-100']:
            x = x.view(x.shape[0], -1).to(torch.float32)
        elif self.data_set in ['COVERTYPE', 'A9A', 'W8A']:
            x = x.to(torch.float32)
        x = self.fc(x)
        return x


'''非凸模型：TinyNN FEMIST and MNIST'''
class TinyNN(nn.Module):
    def __init__(self, args):
        super(TinyNN, self).__init__()
        self.class_num = args.data_distributer.class_num
        self.feature_size = np.prod(args.data_distributer.x_shape)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=200),
            nn.BatchNorm1d(200, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=100),
            nn.BatchNorm1d(100, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=self.class_num)
        )
    def forward(self, x):
        x = x.view(x.shape[0], -1).to(torch.float32)
        x = self.fc(x)
        return x


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


'''非凸模型：VGG-11 cifor-100 确定'''
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        class_num = 100
        self.feature = self.vgg_stack((1, 1, 2, 2, 2),
                                      ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)),
                                      (0.5, 0.5, 0.5, 0.5, 0.5))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            # nn.BatchNorm1d(4096, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, class_num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def vgg_block(self, num_convs, in_channels, out_channels, dropout_p):
        net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels, affine=True),
               nn.ReLU(inplace=True)]

        for i in range(num_convs - 1):
            net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            net.append(nn.BatchNorm2d(out_channels, affine=True))
            net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*net)

    def vgg_stack(self, num_convs, channels, dropout_ps):
        net = []
        for n, c, d in zip(num_convs, channels, dropout_ps):
            in_c = c[0]
            out_c = c[1]
            net.append(self.vgg_block(n, in_c, out_c, d))
        return nn.Sequential(*net)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
