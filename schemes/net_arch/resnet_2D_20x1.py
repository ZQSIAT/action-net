import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['Resnet2D', 'resnet18_2D', 'resnet34_2D', 'resnet50_2D', 'resnet101_2D',
           'resnet152_2D']


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

model_path = {
    'resnet18': '/home/xp_ji/models_resnet/resnet18-5c106cde.pth',
    'resnet34': '/home/xp_ji/models_resnet/resnet34-333f7ec4.pth',
    'resnet50': '/home/xp_ji/models_resnet/resnet50-19c8e357.pth',
    'resnet101': '/home/xp_ji/models_resnet/resnet101-5d3b4d8f.pth',
    'resnet152': '/home/xp_ji/models_resnet/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_segments=16,
                 in_channel=3,
                 dropout=0.5,
                 num_classes=10):
        self.dropout = dropout
        self.inplanes = 64
        self.is_debug = True
        super(ResNet, self).__init__()

        self.conv_0 = nn.Conv2d(in_channel, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn_0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d((3, 1), stride=1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # add dropout by zqs
        if self.dropout == 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Sequential(nn.Dropout(p=self.dropout),
                                    nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input", x.shape)
        x = self.conv_0(x)
        x = self.bn_0(x)
        # print("conv_0", x.shape)
        if self.is_debug:
            print("conv_0", x.shape)
            pass
        x = self.conv1(x)
        # print("conv1", x.shape)
        if self.is_debug:
            print("conv1", x.shape)
            pass
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        # print("maxpool", x.shape)
        if self.is_debug:
            print("maxpool", x.shape)
            pass
        x = self.layer1(x)
        # print("layer1", x.shape)
        if self.is_debug:
            print("layer1", x.shape)
            pass
        x = self.layer2(x)
        # print("layer2", x.shape)
        if self.is_debug:
            print("layer2", x.shape)
            pass
        x = self.layer3(x)
        # print("layer3", x.shape)
        x = self.layer4(x)
        # print("layer4", x.shape)
        if self.is_debug:
            print("layer4", x.shape)
            pass
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.is_debug:
            print("conv2", x.shape)
            pass
        # print("conv2", x.shape)

        x = self.avgpool(x)
        # print("avgpool", x.shape)
        # raise RuntimeError
        x = x.view(x.size(0), -1)
        if self.is_debug:
            print("avgpool view", x.shape)
            pass
        x = self.fc(x)
        if self.is_debug:
            print("fc", x.shape)
            pass
        # print("fc", x.shape)
        # raise RuntimeError
        return x

class Resnet2D(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(Resnet2D, self).__init__()
        self.base_model = ResNet(block, layers, **kwargs)
        self.consensus = ConsensusModule("avg", dim=1)
        self.is_debug = True

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)
        x = x.float()
        if self.is_debug:
            print("begining x", x.shape)
            pass
        out = self.base_model(x)
        if self.is_debug:
            print("out x", out.shape)
            pass
        out = out.view(b, f, -1)
        if self.is_debug:
            print("out.view x", out.shape)
            pass
        out = self.consensus(out)
        if self.is_debug:
            print("consensus x", out.shape)
            pass

        out = out.squeeze(1)

        return out

def resnet18_2D(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet2D(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.base_model.load_state_dict(model_path['resnet18'])
    return model


def resnet34_2D(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet2D(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.base_model.load_state_dict(model_path['resnet34'])
    return model


def resnet50_2D(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet2D(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.base_model.load_state_dict(model_path['resnet50'])
    return model


def resnet101_2D(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet2D(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.base_model.load_state_dict(model_path['resnet101'])
    return model


def resnet152_2D(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet2D(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.base_model.load_state_dict(model_path['resnet152'])

    return model




if __name__ == "__main__":
    import torch

    input = torch.randn((1, 3, 32, 20, 1)).cuda()  # B x C x F x W x H

    model = resnet34_2D().cuda()

    loss_fn = nn.CrossEntropyLoss()
    output = model(input)
    output = output.double()
    target = torch.empty(1, dtype=torch.long).random_(10).cuda()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print("output", output.shape)

    print(res)