import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'PlainNet', 'plain_net_4'
]


class PlainNet(nn.Module):

    def __init__(self,
                 in_channel,
                 sample_size=224,
                 num_segments=16,
                 num_classes=10):

        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, 64, kernel_size=3,
                               stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.conv2= nn.Conv3d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3,
                               stride=(2,2,2), padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=3,
                               stride=(2,2,2), padding=1, bias=False)

        self.bn4 = nn.BatchNorm3d(512)

        last_duration = int(math.ceil(num_segments / 8))
        last_size = int(math.ceil(sample_size / 32))

        self.avgpool = nn.AvgPool3d(
            (last_duration, 2, 3), stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print(x.shape)
        # raise RuntimeError


        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def plain_net_4(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = PlainNet(**kwargs)
    return model


if __name__ == "__main__":
    x = torch.randn((1, 3, 16, 30, 38)).cuda()
    model = plain_net_4(in_channel=3).cuda()
    out = model(x).cuda()
    print(out.shape)