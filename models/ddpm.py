"""models/ddpm.py"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

def get_timestep_embedding(t,channel):
    half = channel // 2
    device = t.device

    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=8, eps=1e-6):
        super().__init__(num_groups, num_channels, eps=eps)

def conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, init_scale=1.0):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
    with torch.no_grad():
        conv.weight.data *= init_scale
    return conv

def nin(in_ch, out_ch, init_scale=1.0):
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
    with torch.no_grad():
        layer.weight.data *= init_scale
    return layer

def linear(in_features, out_features, init_scale=1.0):
    fc = nn.Linear(in_features, out_features)
    with torch.no_grad():
        fc.weight.data *= init_scale
    return fc

class DownsampleBlock(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class UpsampleBlock(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=256, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.dropout = dropout

        self.norm1 = GroupNorm(in_channels)
        self.conv1 = conv2d(in_channels, out_channels)
        self.temb_proj = linear(temb_channels, out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channels, out_channels)

        if in_channels != out_channels:
            self.nin_shortcut = nin(in_channels, out_channels)
        else:
            self.nin_shortcut = None

    def forward(self, x, temb):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)

        h = h + self.temb_proj(swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.q = nin(channels, channels)
        self.k = nin(channels, channels)
        self.v = nin(channels, channels)
        self.proj_out = nin(channels, channels, init_scale=0.0)

    def forward(self, x):
      B, C, H, W = x.shape
      h = self.norm(x)

      q = self.q(h).view(B, C, H*W).permute(0, 2, 1)
      k = self.k(h).view(B, C, H*W)
      v = self.v(h).view(B, C, H*W).permute(0, 2, 1)

      w = torch.bmm(q, k) * (C ** -0.5)
      w = torch.softmax(w, dim=-1)

      h_ = torch.bmm(w, v)
      h_ = h_.permute(0, 2, 1).view(B, C, H, W)

      h_ = self.proj_out(h_)
      return x + h_

class DDPMModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        ch=64,
        ch_mult=(1,2,4),
        num_res_blocks=2,
        attn_resolutions={32},
        dropout=0.0,
        resamp_with_conv=False,
        init_resolution=64
    ):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.num_levels = len(ch_mult)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resamp_with_conv = resamp_with_conv
        self.init_resolution = init_resolution

        self.temb_ch = ch * 4

        self.temb_dense0 = linear(self.ch, self.temb_ch)
        self.temb_dense1 = linear(self.temb_ch, self.temb_ch)

        self.conv_in = conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList()
        curr_ch = ch
        skips_ch = [ch] # <-- 1. 여기서 스킵 커넥션의 채널 수를 기록하기 시작합니다.

        for level, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                layers = [ResnetBlock(curr_ch, out_ch, self.temb_ch, dropout)]
                curr_ch = out_ch
                if (init_resolution // (2 ** level)) in attn_resolutions:
                    layers.append(AttnBlock(curr_ch))
                self.down_blocks.append(nn.ModuleList(layers))
                skips_ch.append(curr_ch) # <-- 특징을 넘길 때 채널 수도 같이 저장
            if level != self.num_levels - 1:
                self.down_blocks.append(DownsampleBlock(curr_ch, resamp_with_conv))
                skips_ch.append(curr_ch)

        self.mid_block = nn.ModuleList([
            ResnetBlock(curr_ch, curr_ch, self.temb_ch, dropout),
            AttnBlock(curr_ch),
            ResnetBlock(curr_ch, curr_ch, self.temb_ch, dropout),
        ])

        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                skip_channel = skips_ch.pop() # <-- 2. 여기서 정확한 채널 수를 꺼내옵니다.
                layers = [ResnetBlock(curr_ch + skip_channel, out_ch, self.temb_ch, dropout)]
                curr_ch = out_ch
                if (init_resolution // (2 ** level)) in attn_resolutions:
                    layers.append(AttnBlock(curr_ch))
                self.up_blocks.append(nn.ModuleList(layers))
            if level != 0:
                self.up_blocks.append(UpsampleBlock(curr_ch, resamp_with_conv))

        self.norm_out = GroupNorm(curr_ch)
        self.conv_out = conv2d(curr_ch, out_channels, kernel_size=3, stride=1, padding=1, init_scale=0.0)

    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = swish(temb)
        temb = self.temb_dense1(temb)

        h = self.conv_in(x)
        skips = [h] # <-- 3. 첫 입력값도 스킵 커넥션에 올바르게 추가

        for block in self.down_blocks:
          if isinstance(block, nn.ModuleList):
              for layer in block:
                  if isinstance(layer, ResnetBlock):
                      h = layer(h, temb)
                  else:
                      h = layer(h)
              skips.append(h)
          else:
              h = block(h)
              skips.append(h)

        for layer in self.mid_block:
          if isinstance(layer, ResnetBlock):
              h = layer(h, temb)
          else:
              h = layer(h)

        for block in self.up_blocks:
          if isinstance(block, nn.ModuleList):
              skip = skips.pop()
              h = torch.cat([h, skip], dim=1) # 저장해둔 특징과 현재 특징을 합침
              for layer in block:
                  if isinstance(layer, ResnetBlock):
                      h = layer(h, temb)
                  else:
                      h = layer(h)
          else:
              h = block(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h