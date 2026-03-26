"""models/progan.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 채널 리스트 / steps
channel_list = [128, 128, 128, 128, 64]
steps = 4

"""WSConv2d"""


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Scale 계산
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # bias는 channel별로 reshape후 더함
        out = self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1)
        return out


"""PixelNorm"""


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        # 각 픽셀마다 벡터의 크기를 1로 정규화
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


"""Up/Down Sampling"""


class UpDownSampling(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        # 최근접 보간
        return F.interpolate(x, scale_factor=self.size, mode='nearest')


"""MinibatchStd"""


class MinibatchStd(nn.Module):
    def forward(self, x):
        bs, _, h, w = x.size()
        # Channel별 픽셀 표준편차 계산 -> 전체 평균 -> (bsx1xhxw) 크기로 복제
        std = torch.std(x, dim=0, keepdim=True).mean().expand(bs, 1, h, w)
        return torch.cat([x, std], dim=1)


"""Generator Block"""


class GeneratorConvBlock(nn.Module):
    def __init__(self, step, scale_size):
        super().__init__()
        self.up = UpDownSampling(scale_size)
        self.conv1 = WSConv2d(channel_list[step - 1], channel_list[step])
        self.conv2 = WSConv2d(channel_list[step], channel_list[step])
        self.lrelu = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.up(x)
        x = self.lrelu(self.conv1(x))
        x = self.pn(x)
        x = self.lrelu(self.conv2(x))
        x = self.pn(x)
        return x


""" Generator Structure"""


class Generator(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

        # --- 초기 블록: 4x4 ---
        self.init = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(128, channel_list[0], 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(channel_list[0], channel_list[0]),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        # --- Progressive Block ---
        self.prog_blocks = nn.ModuleList([GeneratorConvBlock(i + 1, 2) for i in range(steps)])

        # --- toRGB layer ---
        self.toRGB = WSConv2d(channel_list[steps], 3, kernel_size=1, stride=1, padding=0)

    def forward(self, z):
        out = self.init(z)
        for block in self.prog_blocks:
            out = block(out)
        return self.toRGB(out)


""" Discriminator Block """


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.conv1 = WSConv2d(channel_list[step], channel_list[step])
        self.conv2 = WSConv2d(channel_list[step], channel_list[step - 1])
        self.down = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        return self.down(x)


""" Discriminator Structure"""


class Discriminator(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.fromrgb_layers = nn.ModuleList()
        self.prog_blocks = nn.ModuleList()

        for s in range(steps, 0, -1):
            self.fromrgb_layers.append(WSConv2d(3, channel_list[s], kernel_size=1, stride=1, padding=0))
            self.prog_blocks.append(DiscriminatorConvBlock(s))

        self.fromrgb_layers.append(WSConv2d(3, channel_list[0], kernel_size=1, stride=1, padding=0))

        self.prog_blocks.append(nn.Sequential(
            MinibatchStd(),
            WSConv2d(channel_list[0] + 1, channel_list[0], 3, 1, 1),
            nn.LeakyReLU(0.2),
            WSConv2d(channel_list[0], channel_list[0], 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(channel_list[0], 1, 1, 1, 0),
            nn.Sigmoid()
        ))

        self.down = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2)

    def fade_in(self, alpha, down, cur):
        return alpha * cur + (1 - alpha) * down

    def forward(self, x, alpha):
        out = self.lrelu(self.fromrgb_layers[0](x))

        if self.steps == 0:
            return self.prog_blocks[0](out).view(out.size(0), -1)

        down = self.lrelu(self.fromrgb_layers[1](F.avg_pool2d(x, 2)))
        out = self.prog_blocks[0](out)
        out = self.fade_in(alpha, down, out)

        for i in range(1, self.steps + 1):
            out = self.prog_blocks[i](out)

        return out.view(out.size(0), -1)