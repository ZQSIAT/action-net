import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import discriminant_analysis
from sklearn.cross_decomposition import CCA
import time
from multiprocessing import Pool
import multiprocessing
import warnings
import matplotlib.pyplot as plt
from matplotlib import colors

# warnings.filterwarnings('ignore')  # 取消warning输出

__all__ = ['resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class MLP(nn.Module):
    def __init__(self,
                 num_classes=10,
                 dropout=0.5
                 ):
        super(MLP, self).__init__()
        self.is_relu = False
        self.is_cca = True
        self.is_linear = True
        self.is_debug = True
        # self.is_debug = False
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(512, 60)
        self.fc3 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.fusion_dp = nn.Dropout(dropout)

    def forward(self, input1, input2, batch_size):
        #todo: Judging if using linear()
        if self.is_linear:
            input1 = self.fc1(input1)
            input2 = self.fc2(input2)

        # todo: Judging if using relu()?
        if self.is_relu:
            input1 = self.relu(input1)
            input2 = self.relu(input2)
            pass

        if self.is_debug:
            print("input1", input1.shape, "\n", "input2", input2.shape)
            pass

        # todo: Judging if using CCA
        if self.is_cca:
            input1 = input1.cpu().detach().numpy()
            input2 = input2.cpu().detach().numpy()

            cca = CCA(copy=True, max_iter=1024, n_components=50, scale=True, tol=1e-03)
            time1 = time.time()

            cca.fit(input1, input2)
            time2 = time.time()
            input1_c, input2_c = cca.fit_transform(input1, input2)

            # begin plot
            plt.figure()
            row_num = 2
            col_num = 2
            color_list = list(colors.cnames.keys())
            plt.subplot(row_num, col_num, 1)
            plt.plot(input1[:, 0], input1[:, 1], c=color_list[6], marker='.', ls=' ')
            plt.title("input1")
            plt.subplot(row_num, col_num, 2)
            plt.plot(input2[:, 0], input2[:, 1], c=color_list[7], marker='.', ls=' ')
            plt.title('input2')

            plt.subplot(row_num, col_num, 3)
            plt.plot(input1_c[:, 0], input1_c[:, 1], c=color_list[3], marker='.', ls=' ')
            plt.plot(input2_c[:, 0], input2_c[:, 1], c=color_list[9], marker='.', ls=' ')
            plt.title("input1_c")

            plt.subplot(row_num, col_num, 4)
            plt.plot(input2_c[:, 0], input2_c[:, 1], c=color_list[9], marker='.', ls=' ')
            plt.title('input2_c')
            plt.show()

        print('Take time:' + str((time2 - time1)) + 's')

        if self.is_debug:
            print("cca input1_c", input1_c.shape, "\n", "cca input2_c", input2_c.shape)
            pass
        exit()

        x = torch.cat([input1, input2], dim=1)
        if self.is_debug:
            print("cat net1 net2", x.shape)
            pass


        # x = self.fc2(x)
        if self.is_debug:
            print("fc2", x.shape)
            pass
        x = self.fusion_dp(x)
        if self.is_debug:
            print("fusion_dp", x.shape)
            pass
        x = self.fc3(x)
        if self.is_debug:
            print("fc3", x.shape)
            pass
        return x


class Fusion(nn.Module):
    def __init__(self,
                 block1=Bottleneck,
                 block2=BasicBlock,
                 layers=None,
                 num_classes=10,
                 dropout=0.5,
                 in_channel=3,
                 num_segments=32
                 ):
        super(Fusion, self).__init__()
        # self.is_debug = True
        self.is_debug = False
        self.batch_size = 64
        self.net2_inplanes = 64
        self.net2_conv_0 = nn.Conv2d(in_channel, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.net2_bn_0 = nn.BatchNorm2d(64)

        self.net2_conv1 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        self.net2_bn1 = nn.BatchNorm2d(64)
        self.net2_relu1 = nn.ReLU(inplace=True)
        self.net2_maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.net2_res2 = self._make_layer_net2(block2, 64, layers[0], stride=1)
        self.net2_res3 = self._make_layer_net2(block2, 128, layers[1], stride=2)
        self.net2_res4 = self._make_layer_net2(block2, 256, layers[2], stride=2)
        self.net2_res5 = self._make_layer_net2(block2, 512, layers[3], stride=2)
        self.net2_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        self.net2_bn2 = nn.BatchNorm2d(512)
        self.net2_relu2 = nn.ReLU(inplace=True)
        self.net2_avgpool = nn.AvgPool2d((3, 1), stride=1)
        self.net2_fc = nn.Linear(512 * block2.expansion, 512 * block2.expansion)
        self.net2_consensus = ConsensusModule("avg", dim=1)

        self.net1_inplanes = 64  # todo
        self.net1_conv0 = nn.Conv3d(in_channel, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                    bias=False)
        self.net1_bn_0 = nn.BatchNorm3d(64)
        self.net1_conv1 = nn.Conv3d(64, 64, kernel_size=(3, 7, 1), stride=(1, 3, 1), padding=(1, 3, 0), bias=False)
        self.net1_bn1 = nn.BatchNorm3d(64)
        self.net1_relu1 = nn.ReLU(inplace=True)
        self.net1_maxpool = nn.MaxPool3d(kernel_size=(3, 7, 1), stride=(1, 3, 1), padding=(1, 3, 0))
        self.net1_res2 = self._make_layer_net1(block1, 64, layers[0], stride=1)
        self.net1_res3 = self._make_layer_net1(block1, 128, layers[1], stride=2)
        self.net1_res4 = self._make_layer_net1(block1, 256, layers[2], stride=2)
        self.net1_res5 = self._make_layer_net1(block1, 512, layers[3], stride=2)
        self.net1_conv2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 1), stride=(1, 3, 1), padding=(1, 1, 0), bias=False)
        self.net1_bn2 = nn.BatchNorm3d(512)
        self.net1_relu2 = nn.ReLU(inplace=True)
        last_duration = int(math.ceil(num_segments / 8))
        self.net1_avgpool = nn.AvgPool3d((last_duration, 6, 1), stride=1)
        self.net1_fc = nn.Linear(512 * block1.expansion, 512 * block1.expansion)
        self.mlp = MLP(num_classes=num_classes, dropout=dropout)

        self.fusion_dp = nn.Dropout(dropout)
        self.fusion_fc = nn.Linear(1024, num_classes, bias=False)

    def forward(self, input1, input2):
        self.batch_size = input1.size(0)
        net2 = self.net2(input2)
        if self.is_debug:
            print("net2", net2.shape)
            pass
        net1 = self.net1(input1)  # ::16 means that "seq[start:end:step]"
        if self.is_debug:
            print("net1", net1.shape)
            pass

        x = self.mlp.forward(net1, net2, self.batch_size)
        # exit()

        # x = torch.cat([net1, net2], dim=1)
        # if self.is_debug:
        #     print("cat net1 net2", x.shape)
        #     pass
        # x = x.cpu().detach().numpy()
        # pca = PCA(n_components=512)
        # x = pca.fit_transform(x)
        # if self.is_debug:
        #     print("PCA X", x.shape)
        #     pass
        # exit()
        # x = self.fusion_dp(x)
        # if self.is_debug:
        #     print("fusion_dp", x.shape)
        #     pass
        # exit()
        # x = self.fusion_fc(x)
        # if self.is_debug:
        #     print("fusion_fc", x.shape)
        #     pass
        return x

    def net1(self, input):

        x = self.net1_conv0(input)
        if self.is_debug:
            print("net1_conv0", x.shape)
            pass
        x = self.net1_bn_0(x)
        if self.is_debug:
            print("net1_bn_0", x.shape)
            pass
        x = self.net1_conv1(x)
        if self.is_debug:
            print("net1_conv1", x.shape)
            pass
        x = self.net1_bn1(x)
        x = self.net1_relu1(x)
        x = self.net1_maxpool(x)
        if self.is_debug:
            print("net1_maxpool", x.shape)
            pass
        x = self.net1_res2(x)
        if self.is_debug:
            print("net1_res2", x.shape)
            pass
        x = self.net1_res3(x)
        if self.is_debug:
            print("net1_res3", x.shape)
            pass
        x = self.net1_res4(x)
        if self.is_debug:
            print("net1_res4", x.shape)
            pass
        x = self.net1_res5(x)
        if self.is_debug:
            print("net1_res5", x.shape)
            pass
        # exit()
        x = self.net1_conv2(x)
        x = self.net1_bn2(x)
        x = self.net1_relu2(x)
        if self.is_debug:
            print("net1_relu2", x.shape)
            pass
        x = self.net1_avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.net1_fc(x)
        if self.is_debug:
            print("net1_fc", x.shape)
            pass

        return x

    def _make_layer_net1(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.net1_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.net1_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.net1_inplanes, planes, stride, downsample))
        self.net1_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.net1_inplanes, planes))

        return nn.Sequential(*layers)

    def net2(self, input):
        b, c, f, h, w = input.shape
        input = input.contiguous().view(-1, c, h, w)
        if self.is_debug:
            print("net2", input.shape)
            pass
        x = self.net2_conv_0(input)
        if self.is_debug:
            print("net2_conv_0", x.shape)
            pass
        x = self.net2_bn_0(x)
        x = self.net2_conv1(x)
        if self.is_debug:
            print("net2_conv1", x.shape)
            pass
        x = self.net2_bn1(x)
        x = self.net2_relu1(x)
        x = self.net2_maxpool(x)
        if self.is_debug:
            print("net2_maxpool", x.shape)
            pass
        x = self.net2_res2(x)
        if self.is_debug:
            print("net2_res2", x.shape)
            pass
        x = self.net2_res3(x)
        if self.is_debug:
            print("net2_res3", x.shape)
            pass
        x = self.net2_res4(x)
        if self.is_debug:
            print("net2_res4", x.shape)
            pass
        x = self.net2_res5(x)
        if self.is_debug:
            print("net2_res5", x.shape)
            pass
        x = self.net2_conv2(x)
        x = self.net2_bn2(x)
        x = self.net2_relu2(x)
        if self.is_debug:
            print("conv2", x.shape)
            pass
        x = self.net2_avgpool(x)
        x = x.view(x.size(0), -1)
        if self.is_debug:
            print("before net2_fc", x.shape)
            pass
        # exit()
        # x = self.net2_fc(x)
        if self.is_debug:
            print("net2_fc", x.shape)
            pass
        x = x.view(b, f, -1)
        if self.is_debug:
            print("out.view x", x.shape)
            pass
        x = self.net2_consensus(x)
        if self.is_debug:
            print("consensus x", x.shape)
            pass
        x = x.squeeze(1)
        if self.is_debug:
            print("squeeze x", x.shape)
            pass
        return x

    def _make_layer_net2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.net2_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.net2_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.net2_inplanes, planes, stride, downsample))
        self.net2_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.net2_inplanes, planes))

        return nn.Sequential(*layers)


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = Fusion(Bottleneck, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Fusion(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Fusion(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Fusion(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Fusion(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    num_classes = 10
    input_tensor1 = torch.randn((64, 3, 32, 1140, 1)).cuda()
    input_tensor2 = torch.randn((64, 3, 32, 20, 1)).cuda()
    model = resnet18(num_classes=num_classes).cuda()
    output = model(input_tensor1, input_tensor2)

    loss_fn = nn.CrossEntropyLoss()
    output = output.double()
    target = torch.empty(64, dtype=torch.long).random_(10).cuda()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)



