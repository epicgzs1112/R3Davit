# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

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

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, mask)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(torch.nn.Module):
    def __init__(
            self, dim, num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            act_layer=torch.nn.GELU,
            norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, context, mask=None):
        context = context + self.attn(x=self.norm1(context), mask=mask)
        context = context + self.mlp(self.norm2(context))
        return context


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
            BasicR2Plus1D(chan, chan,chan,4,2,3),
            nn.ReLU(),
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
            nn.ReLU(),
            BasicR2Plus1D(chan, chan, chan, 4, 2, 3),
        )

    def forward(self, x):
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
            nn.Conv3d(588, 392, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            ResBlock(392),

            ResBlock(392),

            ResBlock(392), )
        self.uplayer1 = nn.Sequential(
            nn.ConvTranspose3d(392, 196, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.m1 = Mamba(       # x = rearrange(x, '(b v) h w d -> (b v) (h w) d', b=batchsize, v=view,h=7,w=7)  # [B, V*P, D]
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=196,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.uplayer2 = nn.Sequential(
            nn.ConvTranspose3d(196, 98, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(98, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),)
        self.m2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=98,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.uplayer3 = nn.Sequential(
            nn.ConvTranspose3d(98, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.m3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=64,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.flayer=nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))





    def forward(self, x):
        #x = rearrange(x, '(b v) (h w) c-> b v  h w c',h=7,w=7)  # [B, V*P, D]
        #x=x.permute(0,4,1,2,3)
        #print(f'\n x in decoder{x.shape}')
        x=x.reshape(-1,588,4,4,4)
        #print(f'\n decoder:{self.decoder}')
        x=self.layer1(x)
        x=self.uplayer1(x)
        #print(f'\n x after layer1 :{x.shape}')# 8 196 8 8 8
        batchsize = x.shape[0]
        x = rearrange(x, 'b d w h l -> b   (w h l) d', b=batchsize, h=8, w=8,l=8)  # [B, V*P, D]
        x=self.m1(x)
        #print(f'\n x after m1 :{x.shape}')
        x = rearrange(x, ' b   (w h l) d ->b d w h l', b=batchsize, h=8, w=8, l=8)  # [B, V*P, D]
        x=self.uplayer2(x)
        #print(f'\n x after layer2 :{x.shape}')  # 8 98 16 16 16

        x = rearrange(x, 'b d w h l -> b   (w h l) d', b=batchsize, h=16, w=16, l=16)  # [B, V*P, D]
        x = self.m2(x)
        #print(f'\n x after m1 :{x.shape}')
        x = rearrange(x, ' b   (w h l) d ->b d w h l', b=batchsize, h=16, w=16, l=16)  # [B, V*P, D]
        x = self.uplayer3(x)
        #print(f'\n x after layer2 :{x.shape}')  # 8 98 16 16 16

        x = rearrange(x, 'b d w h l -> b   (w h l) d', b=batchsize, h=32, w=32, l=32)  # [B, V*P, D]
        x = self.m3(x)
       # print(f'\n x after m1 :{x.shape}')
        x = rearrange(x, ' b   (w h l) d ->b d w h l', b=batchsize, h=32, w=32, l=32)  # [B, V*P, D]
       # x = self.uplayer4(x)
       # print(f'\n x after layer2 :{x.shape}')  # 8 98 16 16 16
        x=self.flayer(x)
      #  print(f'\n decoder return x :{x.shape}')
        return x


class Transformer(torch.nn.Module):
    def __init__(
            self,
            patch_num=4 ** 3,
            embed_dim=768,
            num_heads=12,
            depth=1,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_num = patch_num
        norm_layer = norm_layer or partial(torch.nn.LayerNorm)  # eps=1e-6
        self.emb = torch.nn.Embedding(patch_num, embed_dim)
        #print(f'\n embed_dim:{embed_dim}')
        self.blocks = torch.nn.ModuleList([
            Blocks(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth)])

    def forward(self, context):
        #print(f'\n in prepare :{context.shape}')
        x = self.emb(torch.arange(self.patch_num, device=context.device))
        #print(f'\n after emb :{x.shape}')
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        #print(f'\n x :{x.shape},context:{context.shape}')
        for blk in self.blocks:
            x = blk(x=x, context=context)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        if cfg.NETWORK.DECODER.VOXEL_SIZE % 4 != 0:
            raise ValueError('voxel_size must be dividable by patch_num')
        if torch.distributed.get_rank() == 0:
            print('Decoder: Progressive Upsampling Transformer-Based')

        #self.patch_num = 4 ** 3
        #self.trans_patch_size = 4
        #self.voxel_size = cfg.NETWORK.DECODER.VOXEL_SIZE
        #self.patch_size = cfg.NETWORK.DECODER.VOXEL_SIZE // self.patch_num

        self.transformer_decoder = TransformerDecoder(
            embed_dim=cfg.NETWORK.DECODER.GROUP.DIM,
            num_heads=cfg.NETWORK.DECODER.GROUP.HEADS,
            depth=cfg.NETWORK.DECODER.GROUP.DEPTH,
            attn_drop=cfg.NETWORK.DECODER.GROUP.SOFTMAX_DROPOUT,
            proj_drop=cfg.NETWORK.DECODER.GROUP.ATTENTION_MLP_DROPOUT)

       # self.prepare = Transformer(embed_dim=1024,num_heads=8)
      #  self.layer_norm = torch.nn.LayerNorm(cfg.NETWORK.MERGER.FC.DIM)

    def forward(self, context):
        # [B, P, D]
        #print(f'\nx in decoder:{context.shape}')
       # context = self.prepare(context=context)
        #print(f'\nx  after prepare:{context.shape}')
     #   context = self.layer_norm(context)
        out = self.transformer_decoder(x=context)# decioder context:torch.Size([16, 64, 768])
       # print(f'\nx after transformer decoder:{out.shape}')#16 1 32 32 32
        return out

