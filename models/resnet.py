"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_mnist(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_mnist, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=8, in_channels=3):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.avgpool = nn.AvgPool2d(8, stride=1)      #artbench10/CIFAR:8, Imagenette:56, EMNIST:7
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print("1",x.shape)
        x = self.avgpool(x)
        #print("2",x.shape)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print("3",x.shape)


        return x

class ResNet_MNIST(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=1):
        super(ResNet_MNIST, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        x = self.fc(x)

        return x

class PreAct_ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_stl10(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        super(ResNet_stl10, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 128)
        # self.fc2 = nn.Linear(128, num_classes)
        self.fc = nn.Linear(256 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc(x)
        return x

def resnet8(num_classes=10, **kwargs):
    model = ResNet_Cifar(BasicBlock, [1, 1, 1], num_classes=num_classes, **kwargs)
    return model


def resnet20(num_classes=8, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes, in_channels=3, **kwargs) #also for GTSRB
    #model = ResNet_stl10(BasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=3, **kwargs) # 18
    #model = ResNet_stl10(BasicBlock, [3, 3, 3], num_classes=num_classes, in_channels=3, **kwargs) # 18
    #model = ResNet_MNIST(BasicBlock, [3, 3, 3], num_classes=num_classes,in_channels=1, **kwargs)

    return model

def resnet20_1(num_classes=8, **kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 3, 3], num_classes=num_classes, in_channels=3, **kwargs) #also for GTSRB
    return model

def resnet20_2(num_classes=8, **kwargs):
    model = ResNet_Cifar(BasicBlock, [4, 3, 3], num_classes=num_classes, in_channels=3, **kwargs) #also for GTSRB
    return model

def resnet20_brain(num_classes=3, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes, in_channels=1, **kwargs)
    return model


def resnet20_mnist(num_classes=10, **kwargs):
    model = ResNet_mnist(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)
    return model


def resnet32(num_classes=10,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [5, 5, 5],num_classes=num_classes, **kwargs)
    #model = ResNet_stl10(BasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=3, **kwargs) #34
    #model = ResNet_stl10(BasicBlock, [5, 5, 5], num_classes=num_classes, in_channels=3, **kwargs) #34

    #model = ResNet_MNIST(BasicBlock, [5, 5, 5], num_classes=num_classes,in_channels=1, **kwargs)

    return model

def resnet32_1(num_classes=10,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [4, 5, 5],num_classes=num_classes, **kwargs)
    return model

def resnet32_2(num_classes=10,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [6, 5, 5],num_classes=num_classes, **kwargs)
    return model

def resnet44(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [7, 7, 7],num_classes=num_classes, **kwargs)
    #model = ResNet_stl10(Bottleneck, [3,4,6,3], num_classes=num_classes, in_channels=3, **kwargs) #50
    #model = ResNet_stl10(BasicBlock, [7, 7, 7], num_classes=num_classes, in_channels=3, **kwargs) # 18


    #model = ResNet_MNIST(BasicBlock, [7, 7, 7], num_classes=num_classes,in_channels=1, **kwargs)
   
    return model

def resnet44_1(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [6, 7, 7],num_classes=num_classes, **kwargs)
    return model

def resnet44_2(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [8, 7, 7],num_classes=num_classes, **kwargs)
    return model


def resnet56(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [9, 9, 9],num_classes=num_classes, **kwargs)
    return model

def resnet56_1(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [8, 9, 9],num_classes=num_classes, **kwargs)
    return model

def resnet56_2(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [10, 9, 9],num_classes=num_classes, **kwargs)
    return model


def resnet62(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [10, 10, 10],num_classes=num_classes, **kwargs)
    return model

def resnet62_1(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [9, 10, 10],num_classes=num_classes, **kwargs)
    return model

def resnet62_2(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [11, 10, 10],num_classes=num_classes, **kwargs)
    return model

# def resnet56(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
#     return model

def resnet86(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [14, 14, 14],num_classes=num_classes, **kwargs)
    return model

def resnet86_1(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [13, 14, 14],num_classes=num_classes, **kwargs)
    return model

def resnet86_2(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [15, 14, 14],num_classes=num_classes, **kwargs)
    return model

def resnet98(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [16, 16, 16],num_classes=num_classes, **kwargs)
    return model

def resnet98_1(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [15, 16, 16],num_classes=num_classes, **kwargs)
    return model

def resnet98_2(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [17, 16, 16],num_classes=num_classes, **kwargs)
    return model

# def resnet110(num_classes=8,**kwargs):
#     print("num_classes",num_classes)
#     model = ResNet_Cifar(BasicBlock, [18, 18, 18],num_classes=num_classes, **kwargs)
#     return model

# def resnet110_1(num_classes=8,**kwargs):
#     print("num_classes",num_classes)
#     model = ResNet_Cifar(BasicBlock, [18, 18, 17],num_classes=num_classes, **kwargs)
#     return model

# def resnet110_2(num_classes=8,**kwargs):
#     print("num_classes",num_classes)
#     model = ResNet_Cifar(BasicBlock, [19, 18, 18],num_classes=num_classes, **kwargs)
#     return model


def resnet110(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [18, 18, 18],num_classes=num_classes, **kwargs)
    return model

def resnet110_1(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [17, 18, 18],num_classes=num_classes, **kwargs)
    return model

def resnet110_2(num_classes=8,**kwargs):
    print("num_classes",num_classes)
    model = ResNet_Cifar(BasicBlock, [19, 18, 18],num_classes=num_classes, **kwargs)
    return model
    

# def resnet110(**kwargs):
#     model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
#     return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == "__main__":
    net = resnet20()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())
