import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, indim, middim, outdim):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(indim, middim, 3)
        self.conv2 = nn.Conv2d(middim, outdim, 3)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(middim)
        self.bn2 = nn.BatchNorm2d(outdim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x2 = self.conv2(x1)
        x = self.relu(self.bn2(x))
        return x


class DownConvBlock(nn.Module):
    def __init__(self, indim, middim, outdim):
        super(DownConvBlock, self).__init__()
        self.conv = ConvBlock(indim, middim, outdim)
        self.pool = nn.Maxpool2d(2, 2)

    def forward(self, x):
        x1 = self.conv(x)
        x = self.pool(x1)
        return x1, x


class UpConvBlock(nn.Module):
    def __init__(self, indim, middim, outdim):
        super(DownConvBlock, self).__init__()
        self.conv = ConvBlock(indim, middim, outdim)

    def forward(self, formalX, x):
        sizeF = formalX.size(3)
        sizeB = x.size(3)
        # 注意sizeF和sizeB为偶数时，成立，否则不成立
        formalX = formalX[:, :, sizeF//2-sizeB//2:sizeF//2+sizeB//2,
                          sizeF//2-sizeB//2:sizeF//2+sizeB//2]
        x = torch.cat([formalX, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
	def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DownConvBlock(1, 64, 64)
        self.down2 = DownConvBlock(64, 128, 128)
        self.down3 = DownConvBlock(128, 256, 256)
        self.down4 = DownConvBlock(256, 512, 512)
        self.bottom = ConvBlock(512, 1024, 512)
        self.up1 = UpConvBlock(1024, 512, 256)
        self.up2 = UpConvBlock(512, 256, 128)
        self.up3 = UpConvBlock(256, 128, 64)
        self.up4 = UpConvBlock(128, 64, 64)
        self.final = nn.Conv2d(64, 2, 1)

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
