import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

__all__ = ['resnet34', 'resnet50', 'resnet101','resnet152', 'resnet200']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1,stride,stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self,
                 block = Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=10,
                 dropout=0.5,
                 in_channel=3,
                 num_segments=3
                 ):
        super(SlowFast, self).__init__()

        self.fast_inplanes = 8
        self.fast_conv0 = nn.Conv3d(3, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.fast_conv1 = nn.Conv3d(8, 8, kernel_size=(5, 3, 1), stride=(1, 1, 1), padding=(2, 1, 0), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=2, head_conv=3)
        
        self.lateral_p1 = nn.Conv3d(8, 8*2, kernel_size=(5, 1, 1), stride=(8,1,1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32,32*2, kernel_size=(5, 1, 1), stride=(8,1,1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64,64*2, kernel_size=(5, 1, 1), stride=(8,1,1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128,128*2, kernel_size=(5, 1, 1), stride=(8,1,1), bias=False, padding=(2, 0, 0))

        self.slow_inplanes = 64+64//8*2
        self.slow_conv0 = nn.Conv3d(3, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.slow_conv1 = nn.Conv3d(64, 64, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes+2048, num_classes, bias=False)
        self.is_debug = False

    def forward(self, input):
        fast, lateral = self.FastPath(input[:, :, ::2, :, :])
        if self.is_debug:
            print("fast", fast.shape)
            print("lateral", len(lateral))
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral)  # ::16 means that "seq[start:end:step]"
        if self.is_debug:
            print("slow", slow.shape)
        x = torch.cat([slow, fast], dim=1)
        # exit()
        x = self.dp(x)
        if self.is_debug:
            print("dp", x.shape)
        x = self.fc(x)
        if self.is_debug:
            print("fc", x.shape)
        return x

    def SlowPath(self, input, lateral):
        x = self.slow_conv0(input)
        if self.is_debug:
            print("slow_conv0", x.shape)
        x = self.slow_conv1(x)
        if self.is_debug:
            print("slow_conv1", x.shape)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        if self.is_debug:
            print("slow_maxpool", x.shape)
        x = torch.cat([x, lateral[0]],dim=1)
        x = self.slow_res2(x)
        if self.is_debug:
            print("slow_maxpool", x.shape)
        x = torch.cat([x, lateral[1]],dim=1)
        x = self.slow_res3(x)
        if self.is_debug:
            print("slow_res3", x.shape)
        x = torch.cat([x, lateral[2]],dim=1)
        x = self.slow_res4(x)
        if self.is_debug:
            print("slow_res4", x.shape)
        x = torch.cat([x, lateral[3]],dim=1)
        x = self.slow_res5(x)
        if self.is_debug:
            print("slow_res5", x.shape)
        # exit()
        x = nn.AdaptiveAvgPool3d(1)(x)
        if self.is_debug:
            print("AdaptiveAvgPool3d", x.shape)
        x = x.view(-1, x.size(1))
        if self.is_debug:
            print("view", x.shape)
        return x

    def FastPath(self, input):
        lateral = []
        x = self.fast_conv0(input)
        if self.is_debug:
            print("fast_conv0", x.shape)
        x = self.fast_conv1(x)
        if self.is_debug:
            print("fast_conv1", x.shape)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        if self.is_debug:
            print("fast_maxpool", pool1.shape)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)
        if self.is_debug:
            print("lateral", len(lateral))
        res2 = self.fast_res2(pool1)
        if self.is_debug:
            print("fast_res2", res2.shape)
        lateral_res2 = self.lateral_res2(res2)
        if self.is_debug:
            print("lateral_res2", lateral_res2.shape)
        lateral.append(lateral_res2)
        if self.is_debug:
            print("lateral", len(lateral))
        res3 = self.fast_res3(res2)
        if self.is_debug:
            print("fast_res3", res3.shape)
        lateral_res3 = self.lateral_res3(res3)
        if self.is_debug:
            print("lateral_res3", lateral_res3.shape)
        lateral.append(lateral_res3)
        if self.is_debug:
            print("lateral", len(lateral))
        res4 = self.fast_res4(res3)
        if self.is_debug:
            print("fast_res4", res4.shape)
        lateral_res4 = self.lateral_res4(res4)
        if self.is_debug:
            print("lateral_res4", lateral_res4.shape)
        lateral.append(lateral_res4)
        if self.is_debug:
            print("lateral", len(lateral))
        res5 = self.fast_res5(res4)
        if self.is_debug:
            print("fast_res5", res5.shape)
        # raise RuntimeError
        x = nn.AdaptiveAvgPool3d(1)(res5)
        if self.is_debug:
            print("AdaptiveAvgPool3d", x.shape)
        x = x.view(-1, x.size(1))
        if self.is_debug:
            print("view", x.shape)
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))
  
        self.slow_inplanes = planes * block.expansion + planes * block.expansion//8*2
        return nn.Sequential(*layers)


def resnet34(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = SlowFast(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == "__main__":
    num_classes = 10
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 32, 20, 1))
    model = resnet34(num_classes=num_classes)
    output = model(input_tensor)
    print(output.size())
