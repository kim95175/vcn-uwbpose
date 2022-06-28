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

class Block2d(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

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
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1.):
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
        #print(self.training)
        x = input + self.drop_path(x)
        
        return x


class ConvNeXt(nn.Module):
    def __init__(self, stack_num=1, frame_skip=1,
                    depths=[3, 3, 18, 3], 
                    #dims=[96, 144, 192, 240],#384, 768], 
                    dims=[96, 192, 384, 576], 
                    drop_path_rate=0., 
                    layer_scale_init_value=1.0, #1e-6, 
                    freeze_backbone=False,
                    feature_size='32'
                 ):
        super().__init__()

        #in_chans = 16*(stack_num//frame_skip)
        in_chans = 64*(stack_num//frame_skip)
        #in_chans = 64
        self.input_d = in_chans
        self.stack_num=stack_num
        self.frame_skip = frame_skip
        self.depths = depths
        self.name = feature_size
        self.num_deconv_filters =[]
        if feature_size == '32':
            dims = [96, 192, 128, 128]
            self.num_deconv_filters =[dims[1]] 
            self.inplanes = dims[1]
            output_dim = 128
        elif feature_size == '128':
            dims = [96, 192, 32, 32]
            self.num_deconv_filters = [32, 32, 32]
            self.inplanes = dims[1]
            output_dim = 32

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=3, stride=3), #kernel_size=3, stride=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            if i < 1:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=1, stride=1),
                )
            else:
                if len(self.num_deconv_filters) > 0 and i == 1:
                    downsample_layer = nn.Sequential(
                        nn.Conv2d(self.num_deconv_filters[-1], dims[i+1], kernel_size=1, stride=1),
                    )
                else:
                    downsample_layer = nn.Sequential(
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=1, stride=1),
                    )
            
            self.downsample_layers.append(downsample_layer)

        

        self.deconv_layers = self._make_deconv_layer(
            len(self.num_deconv_filters),  # NUM_DECONV_LAYERS
            self.num_deconv_filters,  # NUM_DECONV_FILTERS
        ) 

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        #print(f"depth = {sum(depths)} dp_rates = ",len(dp_rates), dp_rates)
        cur = 0
        for i in range(len(dims)):
            if i < 2:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]
            else:
                stage = nn.Sequential(
                    *[Block2d(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]

        self.apply(self._init_weights)
        
  
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=dims[-1], 
                out_channels=output_dim,#=32,  # NUM_JOINTS,
                kernel_size=1,  # FINAL_CONV_KERNEL
                stride=1,
                padding=0  # if FINAL_CONV_KERNEL = 3 else 1
            ),
            LayerNorm2d(output_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU()
       ) 
       
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
            if i == 1:
                b, n, d = x.shape
                root_t = int(d**0.5)
                x = rearrange(x, 'b n (t1 t2) -> b n t1 t2', t1=root_t)
                x = self.deconv_layers(x)
            
        return x

    def forward(self, input):#tensor_list: NestedTensor):

        xs = OrderedDict()
        #input = tensor_list.tensors 
        #print(input.shape) # 128 1 64 78
        x = rearrange(input, 'b t n d -> b (t n) d')
        x = self.forward_features(x)
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
            if i == 1:
                b, n, d = x.shape
                root_t = int(d**0.5)
                x = rearrange(x, 'b n (t1 t2) -> b n t1 t2', t1=root_t)
                x = self.deconv_layers(x)
                print("deconv x ", x.shape) 

        x = self.final_layer(x)
        print("final x ", x.shape) 


    def _make_deconv_layer(self, num_layers, num_filters):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = 4, 1, 0

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(LayerNorm2d(planes, eps=1e-6, data_format="channels_first"))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

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

class LayerNorm2d(nn.Module):
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
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def build_ftr_backbone32(args):
    
    
    backbone = ConvNeXt(stack_num=args.stack_num, 
                            frame_skip=args.frame_skip, 
                            depths=[3,3,9,3], 
                            drop_path_rate=args.drop_prob, 
                            freeze_backbone=args.finetune,
                            feature_size=args.feature)

    return backbone


