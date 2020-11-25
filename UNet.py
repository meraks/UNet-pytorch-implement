import torch
import torch.nn as nn
from torch.nn import functional as F

class DownConv(nn.Module):
    def __init__(self, indim, outdim, isPool=True):
        super(DownConv, self).__init__()
        self.isPool = isPool
        self.conv1 = nn.Conv2d(indim, outdim, 3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.relu(self.bn(x))
        x = self.conv2(x1)
        x2 = self.relu(self.bn(x))
        if self.isPool:
            x = self.pool(x2)
            return x2, x
        else:
            return x

class UpConv(nn.Module):
    def __init__(self, indim, outdim):
        super(UpConv, self).__init__()
        self.conv1 = nn.Conv2d(2*indim, indim, 3, padding=1)
        self.conv2 = nn.Conv2d(indim, outdim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(indim)
        self.bn2 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, formalX, x):
        size = formalX.size(3)
        x = F.interpolate(x, [size, size],
                          mode='bilinear', align_corners=True)
        x = torch.cat([formalX, x], dim=1)
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DownConv(1, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)
        self.bottom = DownConv(512, 512, isPool=False)
        self.up1 = UpConv(512, 256)
        self.up2 = UpConv(256, 128)
        self.up3 = UpConv(128, 64)
        self.up4 = UpConv(64, 64)
        self.final = nn.Conv2d(64, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.bottom(x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        logits = self.final(x)
        return logits



if __name__ == '__main__':
    net = UNet()
    # net = UpConv(512, 256)
    x = torch.Tensor(1, 1, 512, 512)
    # x = torch.Tensor(1, 512, 572, 572)
    # fX = torch.Tensor(1, 512, 64, 64)
    y = net(x)
    print(y.shape)
