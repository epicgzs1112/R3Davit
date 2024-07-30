# -*- coding: utf-8 -*-
#
#ref  UMIFormer: Mining the Correlations between Similar Tokens for Multi-View 3D Reconstruction.
# Developed by chenghuanli <chenghuanli@ecjtu.edu.cn>

import torch
from einops import rearrange
from mamba_ssm import Mamba
from timm.models.vision_transformer import Mlp, partial
from torch import nn

from models.transformer_base.decoder.standard_layers import Block as Blocks
from inspect import isfunction
import numpy as np


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d





class BasicR2Plus1D(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BasicR2Plus1D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, kernel_size, kernel_size),
                               stride=(1, stride, stride), padding=(0, padding, padding), bias=bias)
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                               padding=(1, 0, 0), bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
            nn.ReLU(),
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
            nn.ReLU(),
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
        )

    def forward(self, x):
        # print(f'self.net(x):{self.net(x).shape},x :{x.shape}')
        return self.net(x) + x


class ResBlock3(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
            nn.ReLU(),
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
            nn.ReLU(),
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
        )

    def forward(self, x):
        # print(f'self.net(x):{self.net(x).shape},x :{x.shape}')
        return self.net(x) + x


class ResBlock2(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(chan, chan, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, kernel_size=1),
        )

    def forward(self, x):
        # print(f'self.net(x):{self.net(x).shape},x :{x.shape}')
        return self.net(x) + x


class TransformerDecoder(torch.nn.Module):
    def __init__(
            self,
            patch_num=4 ** 3,
            embed_dim=768,
            num_heads=12,
            depth=8,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=None):
        super().__init__()
        self.patch_num = patch_num
        self.input_side = round(np.power(patch_num, float(1) / float(3)))

        self.layer1 = nn.Sequential(
            BasicR2Plus1D(784, 294, 294, kernel_size=3, padding=1, stride=1),
            ResBlock(294),

        )
        self.layer2 = nn.Sequential(
            BasicR2Plus1D(147, 147, 147, kernel_size=3, padding=1, stride=1),
            ResBlock2(147),

        )
        self.layer3 = nn.Sequential(
            BasicR2Plus1D(64, 64, 64, kernel_size=3, padding=1, stride=1),
            ResBlock2(64),

        )
        self.layer4 = nn.Sequential(
            BasicR2Plus1D(32, 32, 32, kernel_size=3, padding=1, stride=1),
            ResBlock2(32),

        )
        self.uplayer1 = nn.Sequential(
            nn.ConvTranspose3d(294, 147, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(147, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.m1 = Mamba(  # x = rearrange(x, '(b v) h w d -> (b v) (h w) d', b=batchsize, v=view,h=7,w=7)  # [B, V*P, D]
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=147,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=16,  # Block expansion factor
        )
        self.uplayer2 = nn.Sequential(
            nn.ConvTranspose3d(147, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), )
        self.m2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=64,  # Model dimension d_model
            d_state=8,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=8,  # Block expansion factor
        )
        self.uplayer3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.m3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=32,  # Model dimension d_model
            d_state=4,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )
        self.flayer = nn.Conv3d(32, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.layernorm = nn.LayerNorm(1024)

    def forward(self, x):

        x = self.layernorm(x)
        x = x.reshape(-1, 784, 4, 4, 4).contiguous()

        x = self.layer1(x)
        x = self.uplayer1(x)

        batchsize = x.shape[0]
        x = self.layer2(x)
        x = rearrange(x, 'b d w h l -> b   (w h l) d', b=batchsize, h=8, w=8, l=8).contiguous()  # [B, V*P, D]
        x = self.m1(x)


        x = rearrange(x, ' b   (w h l) d ->b d w h l', b=batchsize, h=8, w=8, l=8).contiguous()  # [B, V*P, D]
        x = self.uplayer2(x)

        x = self.layer3(x)
        x = rearrange(x, 'b d w h l -> b   (w h l) d', b=batchsize, h=16, w=16, l=16).contiguous()  # [B, V*P, D]
        x = self.m2(x)


        x = rearrange(x, ' b   (w h l) d ->b d w h l', b=batchsize, h=16, w=16, l=16).contiguous()  # [B, V*P, D]
        x = self.uplayer3(x)

        x = self.layer4(x)
        x = rearrange(x, 'b d w h l -> b   (w h l) d', b=batchsize, h=32, w=32, l=32).contiguous()  # [B, V*P, D]
        x = self.m3(x)


        x = rearrange(x, ' b   (w h l) d ->b d w h l', b=batchsize, h=32, w=32, l=32).contiguous()  # [B, V*P, D]


        x = self.flayer(x)

        return x





class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        if cfg.NETWORK.DECODER.VOXEL_SIZE % 4 != 0:
            raise ValueError('voxel_size must be dividable by patch_num')
        if torch.distributed.get_rank() == 0:
            print('Decoder: Progressive Upsampling Transformer-Based')



        self.transformer_decoder = TransformerDecoder(
            embed_dim=cfg.NETWORK.DECODER.GROUP.DIM,
            num_heads=cfg.NETWORK.DECODER.GROUP.HEADS,
            depth=cfg.NETWORK.DECODER.GROUP.DEPTH,
            attn_drop=cfg.NETWORK.DECODER.GROUP.SOFTMAX_DROPOUT,
            proj_drop=cfg.NETWORK.DECODER.GROUP.ATTENTION_MLP_DROPOUT)



    def forward(self, context):

        out = self.transformer_decoder(x=context)

        return out

