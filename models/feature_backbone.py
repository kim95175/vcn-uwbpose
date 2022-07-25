# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import math
from math import exp
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
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
        x = input + self.drop_path(x)
        
        return x


class ConvNeXt(nn.Module):
    def __init__(self, stack_num=1, frame_skip=1,
                    depths=[3, 3, 12, 3], 
                    #dims=[96, 144, 192, 240],#384, 768], 
                    dims=[96, 192, 384, 768], 
                    drop_path_rate=0., 
                    drop_prob = 0.2, #0.2,
                    drop_size = 4,
                    num_txrx = 8,
                    feature_list = [16],
                    freeze_ftr_backbone = False,
                    layer_scale_init_value=1.0, #1e-6, 
                 ):
        super().__init__()

        #in_chans = 16*(stack_num//frame_skip)
        in_chans = (num_txrx**2)*(stack_num//frame_skip)
        #in_chans = 64
        self.input_d = in_chans
        self.stack_num=stack_num
        self.frame_skip = frame_skip
        self.depths = depths
        self.feature_list = feature_list
        self.mlp = False#True
        
        self.freeze_ftr_backbone = freeze_ftr_backbone

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=3, stride=3), #kernel_size=3, stride=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            if self.mlp and i == 1:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=1, stride=1),
                )
            
            self.downsample_layers.append(downsample_layer)

        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        #print(f"depth = {sum(depths)} dp_rates = ",len(dp_rates), dp_rates)
        #print(f"use selayer =", selayer)
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

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
        
        
        self.num_channels = dims[-1]#//2
        self.mlp_head = None
        final_chan = 256
        if 32 in self.feature_list:
            dim = 16*16
            hidden_featuremap_dim = 16*32
            featuremap_dim = 32*32
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_featuremap_dim),
                nn.LayerNorm(hidden_featuremap_dim),
                nn.Linear(hidden_featuremap_dim, featuremap_dim)
            )
            final_chan = 128
        

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(dims[-1], momentum=0.1),
            nn.Conv2d(
                in_channels=dims[-1], #64,#dims[-1], #self.num_deconv_filters[-1],  # NUM_DECONV_FILTERS[-1]
                out_channels=final_chan,#=32,  # NUM_JOINTS,
                kernel_size=1,  # FINAL_CONV_KERNEL
                stride=1,
                padding=0  # if FINAL_CONV_KERNEL = 3 else 1
            ),
       ) 

        if freeze_ftr_backbone:
            print("@@@@@@@freeze ftr_backbone part@@@@@@@@@@")
            for p in self.parameters():
                p.requires_grad_(False)

                
        self.check_dim()
        #n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print('feature_backbone = number of params:', n_parameters)

        


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            #print(m, m.weight, m.bias)
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i==2 or i==3:
                if self.drop_block is not None:
                    x = self.drop_block(x)
            
        return x

    def forward(self, input):#tensor_list: NestedTensor):
        if self.drop_block is not None:
            self.drop_block.step()
        xs = OrderedDict()
        #input = tensor_list.tensors 
        #print(input.shape) # 128 1 64 78
        x = rearrange(input, 'b t n d -> b (t n) d')
        x = self.forward_features(x)

        if self.mlp_head is not None:
            x = self.mlp_head(x)
        b, n, d = x.shape
        root_t = int(d**0.5)
        x = rearrange(x, 'b n (t1 t2) -> b n t1 t2', t1=root_t)
        
        x = self.final_layer(x)


        out = {'pred_feature' : x }
        return out
    
    def check_dim(self):
        print("____check dimension in Feature Backbone_____")

        #x = torch.zeros((128, self.input_d, 1536))
        #x = torch.zeros((128, self.input_d, 1024)) 
        x = torch.zeros((128, self.input_d, 768)) 
        print("input = ", x.shape)
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            print(f"{i} downsample layer = {x.shape}")
            x = self.stages[i](x)
            print(f"{i} stage = {x.shape} x {self.depths[i]}")

        if self.mlp_head is not None:
            x = self.mlp_head(x)
            print("mlp_head x ", x.shape) 
        b, n, d = x.shape
        root_t = int(d**0.5)
        x = rearrange(x, 'b n (t1 t2) -> b n t1 t2', t1=root_t)
        #print(f"Final x = {x.shape}")

        x = self.final_layer(x)
        print("final x ", x.shape) 


    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 5:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding



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



def build_ftr_backbone(args):
    
    
    backbone = ConvNeXt(stack_num=args.stack_num, 
                            frame_skip=args.frame_skip, 
                            depths=[3,3,18,3], 
                            drop_path_rate=args.drop_prob, 
                            drop_prob=args.dropblock_prob,
                            drop_size=args.drop_size,
                            num_txrx =args.num_txrx,
                            feature_list = args.feature,
                            freeze_ftr_backbone = (args.frozen_ftr_weights is not None)
    )

    return backbone
