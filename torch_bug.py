import torch
a = torch.randn(1, 2048, 8, 11, 11).cuda()
conv = torch.nn.Conv3d(2048, 512, (3, 1, 1), stride=(1, 1, 1)).cuda()
out = conv(a)
b = torch.randn(1, 2048, 8, 12, 12).cuda()
