# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, einsum
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import random
from torch.nn import BatchNorm1d as Normlayer
torch.backends.cudnn.benchmark = True
#from torch.nn import LayerNorm

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class DropBlock1D(nn.Module):
    def __init__(self, drop_prob = 0.1, block_size = 16):
        super(DropBlock1D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size= block_size
        #print(self.block_size // 2)

    def __repr__(self):
        return f'DropBlock1D(drop_prob={self.drop_prob}, block_size={self.block_size})'

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            #print("block_mask", block_mask[:, None, :].shape)
            #print(block_mask.shape, block_mask[0])
            out = x * block_mask[:, None, :]
            out = x * block_mask.numel() / block_mask.sum()
        
        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(input=mask[:, None, :],
                            kernel_size=self.block_size, 
                            stride=1,
                            padding=self.block_size // 2)
        if self.block_size % 2 == 0: 
            block_mask = block_mask[:, :, :-1]
        
        block_mask = 1 - block_mask.squeeze(1)
        
        return block_mask

    def _compute_gamma(self, x):
        #return self.drop_prob / (self.block_size)
        return self.drop_prob / (self.block_size ** 2)

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        #self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        #print("gamma = ", self.gamma)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        #print(self.training)
        x = input + self.drop_path(x)
        
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stack_num=16, frame_skip=4,
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 drop_prob = 0.2, #0.2,
                 drop_size = 4,
                 num_txrx = 8,
                 layer_scale_init_value=1.0, #1e-6, 
                 ):
        super().__init__()


        in_chans = (num_txrx*num_txrx)*(stack_num//frame_skip)
        if stack_num//frame_skip > 1:
            dims[0] = 192
        self.input_d = in_chans
        self.stack_num=stack_num
        self.frame_skip = frame_skip
        self.depths = depths

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=3, stride=3),
            #nn.Conv1d(in_chans, dims[0], kernel_size=1, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            if False:
            #if i == 1:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=3),
                )
            else:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv1d(dims[i], dims[i+1], kernel_size=1, stride=1),
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print(f"depth = {sum(depths)} dp_rates = ",len(dp_rates), dp_rates)
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.apply(self._init_weights)

        if drop_prob > 0.:
            self.drop_block = LinearScheduler(
                dropblock = DropBlock1D(block_size=drop_size, drop_prob=0.),
                start_value=0.,
                stop_value =drop_prob,
                nr_steps=2e4 #5e3
            )
            print(self.drop_block.dropblock)
        else:
            self.drop_block = None
        #print(self.downsample_layers)
        #print(self.stages)
        self.check_dim()
        self.num_channels = dims[-1]
        
        freeze_backbone = False
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i==2 or i==3:
                if self.drop_block is not None:
                    x = self.drop_block(x)
        return x #self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, input_tensor): #tensor_list: NestedTensor):
        #xs = OrderedDict()
        #input = tensor_list.tensors
        if self.drop_block is not None:
            self.drop_block.step()
        x = rearrange(input_tensor, 'b t n d -> b (t n) d')
        x = self.forward_features(x)

        b, n, d = x.shape
        root_t = int(d**0.5)
        x = rearrange(x, 'b n (t1 t2) -> b n t1 t2', t1=root_t)

        mask = torch.zeros((b, root_t, root_t), dtype=torch.bool, device=input_tensor.device)
        return x, mask, None
    
        
    def check_dim(self):
        print("____check dimension in ConvNeXtBackbone6_____")        
        x = torch.zeros((128, self.input_d, 768))
        print("input = ", x.shape)
        for i in range(4):
            x = self.downsample_layers[i](x)
            print(f"{i} downsample layer = {x.shape}")
            x = self.stages[i](x)
            print(f"{i} stage = {x.shape} x {self.depths[i]}")




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.stack_num = backbone.stack_num
        self.frame_skip = backbone.frame_skip
        self.num_channels = backbone.num_channels

    def forward(self, tensor):
        src, mask, _ = self[0](tensor)
        pos = self[1](src, mask).to(src.dtype)
        
        return src, mask, pos

def build_convbackbone(args):

    backbone = ConvNeXt(stack_num=args.stack_num, 
                    frame_skip=args.frame_skip, 
                    depths=[3,3,args.res4dim,3], 
                    drop_path_rate=args.drop_prob,
                    drop_prob=args.dropblock_prob,
                    drop_size=args.drop_size,
                    num_txrx =args.num_txrx)
    #return backbone
    if args.position_embedding is not 'none':
        position_embedding = build_position_encoding(args)
        model = Joiner(backbone, position_embedding)
        return model
    
    return backbone


