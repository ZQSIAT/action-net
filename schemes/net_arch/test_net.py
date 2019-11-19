import torch
from torch import nn
import torch.nn.functional as F


class TestNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=64,
                               kernel_size=(1, 7, 7), padding=(0, 3, 3), stride=(1, 2, 2)) # TxHxW

        self.conv2 = nn.Conv3d(in_channels=64,
                               out_channels=128,
                               kernel_size=(1, 7, 7), padding=(0, 3, 3), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(in_channels=128,
                               out_channels=256,
                               kernel_size=(1, 7, 7), padding=(0, 3, 3), stride=(1, 2, 2))

        self.conv4 = nn.Conv3d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 7, 7), padding=(0, 3, 3), stride=(1, 2, 2))

        self.conv5 = nn.Conv3d(in_channels=512,
                               out_channels=512,
                               kernel_size=(7, 7, 7), padding=(3, 3, 3), stride=(2, 2, 2))

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 7, 7))

        self.relu = nn.ReLU()

        self.fc = nn.Linear(512, num_classes)

        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x): #16x3x8x224x224
        out = self.conv1(x) #16x64x8x112x112
        out = self.relu(out)

        out = self.conv2(out) #16x128x8x56x56
        out = self.relu(out)

        out = self.conv3(out) #16x256x8x28x28
        out = self.relu(out)

        out = self.conv4(out) #16x512x8x14x14
        out = self.relu(out)

        out = self.conv5(out)  # 16x512x4x7x7
        out = self.relu(out)

        out = self.maxpool(out) # 16x512x1x1x1

        out = out.view(x.size(0), -1)

        out = self.fc(out)


        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return out


if __name__ == "__main__":

    input = torch.randn((1, 3, 8, 224, 224)).cuda() # BxCxTxHxW

    print(input.shape)

    model = TestNet(in_channels=3, num_classes=100).cuda()
    print(model)

    output = model(input)
    print(output.shape)



