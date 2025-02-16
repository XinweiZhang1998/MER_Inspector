"""
Code adapted from https://github.com/xternalz/WideResNet-pytorch
Modifications = return activations for use in attention transfer,
as done before e.g in https://github.com/BayesWatch/pytorch-moonshine
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Track Running Stats #DBG
trs = True


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=trs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=trs)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        dropRate=0.0,
        upsample=False,
        in_channels=3,
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # Upsample in input is smaller than 32x32  STL-96,96
        self.upsample = None
        if upsample:
            self.upsample = torch.nn.Upsample(
                size=(32, 32), mode="bilinear", align_corners=False
            )

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], track_running_stats=trs)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.conv1(x)
        #print("1",out.shape)
        out = self.block1(out)
        out = self.block2(out)
        #print("2",out.shape)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        #print("3",out.shape)
        out =self.pool(out)
        #out = F.avg_pool2d(out, 56) #CIFAR10, CIFAR100:8, STL-10:24, Imagenet: 56
        #print("4",out.shape)
        out = out.view(-1, self.nChannels)
        #print("5",out.shape)
        out = self.fc(out)
        #print("6",out.shape)
        return out

if __name__ == "__main__":
    import random
    import time

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    ### WideResNets
    # Notation: W-depth-widening_factor
    # model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.0)
    # model = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    # model = WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.0)
    # model = WideResNet(depth=16, num_classes=10, widen_factor=10, dropRate=0.0)
    # model = WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.0)
    # model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
    # model = WideResNet(depth=40, num_classes=10, widen_factor=10, dropRate=0.0)
    # model = WideResNet(depth=40, num_classes=10, widen_factor=1, dropRate=0.0)
    model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    ###model = WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0)

    t0 = time.time()
    output, *act = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHPAE: ", output.shape)

