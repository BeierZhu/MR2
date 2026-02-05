'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'preact20', 'preact32', 'preact44', 'preact56', 'preact110']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)





class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes,
                                      kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, use_norm=False):
        super(PreActResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        # self.linear = nn.Linear(64 * block.expansion, num_classes)
        if use_norm:
            self.linear = NormedLinear(64 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        return self.linear(out), out
    

def preact20(num_classes=100, use_norm=False):
    return PreActResNet(PreActBlock, [3, 3, 3], num_classes=num_classes, use_norm=use_norm)


def preact32(num_classes=100, use_norm=False):
    return PreActResNet(PreActBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def preact44():
    return PreActResNet(PreActBlock, [7, 7, 7])


def preact56(num_classes=100, use_norm=False):
    return PreActResNet(PreActBlock, [9, 9, 9], num_classes=num_classes, use_norm=use_norm)


def preact110():
    return PreActResNet(PreActBlock, [18, 18, 18])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))



class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    Wide ResNet for CIFAR-10/100
    Commonly used variants:
        WRN-28-10  -> depth=28, widen_factor=10
        WRN-28-2   -> depth=28, widen_factor=2
        WRN-16-8   -> depth=16, widen_factor=8
        ...
    """
    def __init__(self, depth, widen_factor, num_classes=100,
                 dropout_rate=0.3, use_norm=False):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n + 4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, stride=1,
                                       dropout=dropout_rate)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, stride=2,
                                       dropout=dropout_rate)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, stride=2,
                                       dropout=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)

        if use_norm:
            self.linear = NormedLinear(nStages[3], num_classes)
        else:
            self.linear = nn.Linear(nStages[3], num_classes)

        self.apply(_weights_init)

    def _wide_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        features = out.view(out.size(0), -1)
        logits = self.linear(features)
        return logits, features


# 常用配置
def wrn28_10(num_classes=100, dropout_rate=0.3, use_norm=False):
    """WRN-28-10, ~36.5M params (standard CIFAR version)"""
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes,
                      dropout_rate=dropout_rate, use_norm=use_norm)


def wrn28_2(num_classes=100, dropout_rate=0.0, use_norm=False):
    """WRN-28-2, ~2.2M params"""
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes,
                      dropout_rate=dropout_rate, use_norm=use_norm)


def wrn16_8(num_classes=100, dropout_rate=0.3, use_norm=False):
    """WRN-16-8, ~11M params"""
    return WideResNet(depth=16, widen_factor=8, num_classes=num_classes,
                      dropout_rate=dropout_rate, use_norm=use_norm)


def wrn40_2(num_classes=100, dropout_rate=0.3, use_norm=False):
    """WRN-40-2"""
    return WideResNet(depth=40, widen_factor=2, num_classes=num_classes,
                      dropout_rate=dropout_rate, use_norm=use_norm)


# 更新 __all__ （记得把原来的 __all__ 加上这些新名字）
__all__ += ['wrn28_10', 'wrn28_2', 'wrn16_8', 'wrn40_2', 'WideResNet']


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()