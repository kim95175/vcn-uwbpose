# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torch import Tensor
from PIL import Image
import random
import cv2
import numpy as np
import os
from einops import rearrange, repeat


import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list


try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

BN_MOMENTUM = 0.1

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.num_layers = 3

        h = [hidden_dim, hidden_dim//2]        
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        #self.layers = nn.Sequential(
        #                nn.Linear(input_dim, h[0]),
        #                nn.GELU(),
        #                nn.Linear(h[0], h[1]),
        #                nn.GELU(),
        #                nn.Linear(h[1], output_dim),
        #            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

class RFTRpose(nn.Module):
    def __init__(self, rftr, freeze_rftr=False, method='simdr', 
                dr_size=256, ftr_backbone=None, feature_list=[0], roi=False):
        super().__init__()
        print("_______RFTRpose Init________")
        self.rftr = rftr

        if freeze_rftr:
            print("@@@@@@@freeze rftr part@@@@@@@@@@")
            for p in self.parameters():
                p.requires_grad_(False)
        
        self.method = method
        
        self.output_size = dr_size
        self.num_joints = 13

        self.ftr_backbone = ftr_backbone
        self.num_queries = self.rftr.num_queries

        self.use_feature = True if ftr_backbone != None else False
        
        self.box_feature = rftr.box_feature
        self.feature_list = feature_list
        if self.use_feature:
            self.freeze_ftr_backbone = self.ftr_backbone.freeze_ftr_backbone
            if self.freeze_ftr_backbone:
                self.feature_list = [16]

        print("feature_list = ", self.feature_list)

        hidden_dim, nheads = rftr.transformer.d_model, rftr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        
        self.roi = roi
        if self.roi:
            self.ftr_size = 16
            self.roi_output_size = 4 #8
            roi_hidden_dim = self.ftr_size**2 #(self.roi_output_size*2)**2
            print(f"roi mlp dim = {self.roi_output_size**2} -> {roi_hidden_dim} -> {self.ftr_size**2}")
            self.roi_head = nn.Sequential(
                nn.LayerNorm(self.roi_output_size**2),
                nn.Linear(self.roi_output_size**2, roi_hidden_dim),
                nn.LayerNorm(roi_hidden_dim),
                nn.Linear(roi_hidden_dim, self.ftr_size**2)
            )
        
        self.pose_head = PoseHeadSmallConv(hidden_dim + nheads, hidden_dim, self.num_joints, 
                            method, use_feature=self.feature_list)
           
        self.hm_shape = 64
        self.only_feature = False
        self.num_txrx = self.rftr.num_txrx

        
        self.final_dr_layer = nn.Conv2d(
            in_channels=self.num_joints, 
            out_channels=self.num_joints,  
            kernel_size=2,#1,
            stride=2, #2 # 1
            padding=0  # if FINAL_CONV_KERNEL = 3 else 1
        )
        mlp_dim = 32*32
        hidden_dr_dim = mlp_dim // 2 
        output_dim = dr_size
        self.output_dim = output_dim
        self.mlp_head_x = MLP(mlp_dim, hidden_dr_dim, output_dim, 3)
        self.mlp_head_y = MLP(mlp_dim, hidden_dr_dim, output_dim, 3)
        
        
        dummy_input = torch.zeros((4, int(self.rftr.stack_num//self.rftr.frame_skip), self.num_txrx**2, 768))
        self.check_dim(dummy_input)

    #def forward(self, samples: NestedTensor, features: Tensor):
    def forward(self, samples, targets=None):
        
        src, mask, pos, vc = self.rftr.backbone(samples)
        bs = src.shape[0]
        assert mask is not None

        src_flat = src.flatten(2)
        src_proj = self.rftr.input_proj(src_flat)
        src_proj = rearrange(src_proj, 'b n (t1 t2) -> b n t1 t2', t1=16)
        features = [None, None, None]

        if self.use_feature:
            if self.only_feature:
                if 16 in self.feature_list:
                    tgt_features = torch.stack([t["features"] for t in targets])
                    #tgt_features, _ = nested_tensor_from_tensor_list(tgt_features).decompose()
                    tgt_features = tgt_features.to(src_proj)
                    features[0] = tgt_features
                
                if 32 in self.feature_list:
                    tgt_features32 = torch.stack([t["features32"] for t in targets])
                    #tgt_features32, _ = nested_tensor_from_tensor_list(tgt_features32).decompose()
                    tgt_features32 = tgt_features32.to(src_proj)
                    features[1] = tgt_features32

            else:
                proj_features = self.ftr_backbone(samples)['pred_feature']
                if 16 in self.feature_list:
                    features[0] = proj_features
     
        hs, memory = self.rftr.transformer(src_proj, mask, self.rftr.query_embed.weight, pos)

        hs = hs[-1]
        outputs_class = self.rftr.class_embed(hs)
        outputs_coord = self.rftr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
        
        
        if self.rftr.aux_loss:
            out['aux_outputs'] = self.rftr._set_aux_loss(outputs_class, outputs_coord)
        

        if 0 not in self.feature_list:   #if self.use_feature:
            if 16 in self.feature_list:
                out['pred_feature'] = features[0]
            if 32 in self.feature_list:
                out['pred_feature32'] = features[1]

            bbox_mask = self.bbox_attention(hs, memory, mask=mask)
            #print(features[1].shape)
            seg_pose = self.pose_head(src_proj, bbox_mask, features)

        else:
            bbox_mask = self.bbox_attention(hs, memory, mask=mask)
            seg_pose = self.pose_head(src_proj, bbox_mask)


        if self.method =='hm':
            outputs_pose_masks = seg_pose.view(bs, self.rftr.num_queries, self.num_joints, self.hm_shape, self.hm_shape)
            out["pred_hm"] = outputs_pose_masks
        elif self.method =='hmdr' or self.method =='simdr':
            out["pred_hm"] = seg_pose.view(bs, self.rftr.num_queries, self.num_joints, self.hm_shape, self.hm_shape)
            seg_pose = self.final_dr_layer(seg_pose)
            seg_pose = seg_pose.flatten(2) #if not self.no_dec else seg_pose.flatten(3)
            outputs_x = self.mlp_head_x(seg_pose)
            outputs_y = self.mlp_head_y(seg_pose)
            outputs_x = outputs_x.view(bs, self.rftr.num_queries, self.num_joints, self.output_size)
            outputs_y = outputs_y.view(bs, self.rftr.num_queries, self.num_joints, self.output_size)
            out['x_coord'] = outputs_x
            out['y_coord'] = outputs_y
            out['output_size'] = self.output_size

        return out
    
    
    def check_dim(self, samples: NestedTensor):

        print("____check dimension in RFTRPose_____")
        src, mask, pos, vc = self.rftr.backbone(samples)
        print(f"features[-1].decompose() = src : {src.shape}, mask : {mask.shape}")
        bs = src.shape[0]
        assert mask is not None
        features = [None, None, None] 

        src_flat = src.flatten(2)
        src_proj = self.rftr.input_proj(src_flat)
        src_proj = rearrange(src_proj, 'b n (t1 t2) -> b n t1 t2', t1=16)
        print(f"src_proj = {src_proj.shape}")
        if vc is not None:
            vc_query = self.rftr.vc_input_proj(vc)
            print(f"input_proj[{src_proj.shape}] + vc_query[{vc_query.shape}]" )
            src_proj = torch.cat((src_proj, vc_query), dim=1)
            print(f"self.input_proj(src) : {src_proj.shape}")

        if self.use_feature:
            if self.only_feature:
                if 16 in self.feature_list:
                    tgt_features = torch.zeros((4, 256, 16, 16))
                    features[0] = tgt_features.to(src_proj)
                    print("tgt_features.shape = ", features[0].shape)
                if 32 in self.feature_list:
                    tgt_features32 = torch.zeros((4, 128, 32, 32))
                    features[1] = tgt_features32.to(src_proj)
                    print("tgt_features.shape = ", features[1].shape)
            
            else:
                proj_features = self.ftr_backbone(samples)['pred_feature']
                if 16 in self.feature_list:
                    features[0] = proj_features
                    print("features.shape = ", features[0].shape)

        
        hs, memory = self.rftr.transformer(src_proj, mask, self.rftr.query_embed.weight, pos) 
        print(f"hs(decoder) = {hs.shape}, memory(encoder) = {memory.shape}")

        hs = hs[-1]
        outputs_class = self.rftr.class_embed(hs)
        outputs_coord = self.rftr.bbox_embed(hs).sigmoid()
        print("after ffn ", outputs_class.shape, outputs_coord.shape)
        out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}

        # FIXME h_boxes takes the last one computed, keep this in mind
        print(f"bbox_attention input new_hs= {hs.shape}, memory_backbone = {memory.shape}")
        
        if self.rftr.aux_loss:
            out['aux_outputs'] = self.rftr._set_aux_loss(outputs_class, outputs_coord)
        
        if vc is not None:
            print("visual clue trained in box prediction = ", vc.shape)
            features = vc
        
        if features != None: #self.use_feature:
            
            if 16 in self.feature_list:
                out['pred_feature'] = features[0]
                print("[16]features input to the pose_head = ", features[0].shape)
            if 32 in self.feature_list:
                out['pred_feature32'] = features[1]
                print("[32]features input to the pose_head = ", features[1].shape)
                
            if False: #self.roi:
                boxes = box_ops.box_cxcywh_to_xyxy(outputs_coord)
                features = _expand(features, boxes.shape[1])
                boxes = boxes.flatten(0, 1)
                print(boxes.shape, features.shape)
                ids = torch.cat(
                    [
                        torch.full((1,1), i, dtype=boxes.dtype, layout=torch.strided, device=boxes.device) 
                        for i in range(len(list(boxes)))
                    ],
                    dim=0,
                )
                boxes = torch.mul(boxes, self.ftr_size)
                rois = torch.cat([ids, boxes], dim=1)
                roi_features = roi_align(features, rois, output_size=(self.roi_output_size,self.roi_output_size), \
                                            spatial_scale=1.0, aligned=True)
                print("roi_featurs =", roi_features.shape)
                roi_features = roi_features.flatten(2)
                roi_features = self.roi_head(roi_features)
                roi_features = rearrange(roi_features,'b c (p1 p2) -> b c p1 p2',p1=self.ftr_size,p2=self.ftr_size)
                print("roi_featurs =", roi_features.shape)
                bbox_mask = self.bbox_attention(hs, memory, mask=mask)
                seg_pose = self.pose_head(src_proj, bbox_mask, roi_features)
            else:
                bbox_mask = self.bbox_attention(hs, memory, mask=mask)
                seg_pose = self.pose_head(src_proj, bbox_mask, features)
        else:
            bbox_mask = self.bbox_attention(hs, memory, mask=mask)
            seg_pose = self.pose_head(src_proj, bbox_mask)

        print(f"bbox_mask = {bbox_mask.shape}")
        print(f"seg_pose = {seg_pose.shape}")

        if self.method =='hm':
            outputs_pose_masks = seg_pose.view(bs, self.rftr.num_queries, self.num_joints, self.hm_shape, self.hm_shape)
            out["pred_hm"] = outputs_pose_masks
            print("pose hm outputs= ", outputs_pose_masks.shape)
        elif self.method =='hmdr' or self.method == 'simdr':
            out["pred_hm"] = seg_pose.view(bs, self.rftr.num_queries, self.num_joints, self.hm_shape, self.hm_shape)
            print("pose hm outputs= ", out["pred_hm"].shape)
            seg_pose = self.final_dr_layer(seg_pose)
            print("final dr layer ", seg_pose.shape)
            seg_pose = seg_pose.flatten(2) #if not self.no_dec else seg_pose.flatten(3)
            outputs_x = self.mlp_head_x(seg_pose)
            outputs_y = self.mlp_head_y(seg_pose)
            print("pose dr outputs= ", outputs_x.shape, outputs_y.shape)
            outputs_x = outputs_x.view(bs, self.rftr.num_queries, self.num_joints, self.output_size)
            outputs_y = outputs_y.view(bs, self.rftr.num_queries, self.num_joints, self.output_size)
            print("pose dr outputs[x, y]= ", outputs_x.shape, outputs_y.shape)
            out['x_coord'] = outputs_x
            out['y_coord'] = outputs_y
            out['output_size'] = self.output_size

        print("outputs = ", out.keys())
        return out


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

def _expand1d(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1).flatten(0, 1)


class PoseHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, context_dim, final_channel, method='simdr', use_feature=[0]):
        super().__init__()
        

        self.use_feature = use_feature
        self.method = method
        self.dim = dim
        self.heatmap_size = [64, 64]
        
        layer_dim = 16*16 
        inter_dims =[context_dim//2, context_dim // 4, context_dim // 4]   
        if context_dim == 512:
            inter_dims =[context_dim//2, context_dim // 4, context_dim // 8]   

        if 0 not in use_feature:
            #inter_dims =[context_dim //2, context_dim // 4 , context_dim // 4 ]  # 128, 64  64
            inter_dims =[context_dim , context_dim // 2 , context_dim // 2]  # hmdr3  256 128 128
            
            print("PoseHead feature size ", use_feature)
            if 16 in use_feature:
                dim_add = 128
                self.adapter = torch.nn.Conv2d(256, dim_add, 1)
            if 32 in use_feature:
                dim_add32 = 64
                self.adapter32 = torch.nn.Conv2d(128, dim_add32, 1)
                self.mlp_head32 = nn.Sequential(
                    nn.LayerNorm(layer_dim),
                    nn.Linear(layer_dim, 16*32),
                    nn.LayerNorm(16*32),
                    nn.Linear(16*32, 32*32)
                )
 

        print("PoseHead Inter_dims ", inter_dims)
        
        self.lay1 = torch.nn.Conv2d(dim, inter_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[0])
        self.lay2 = torch.nn.Conv2d(inter_dims[0], inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])

        #self.lay3 = torch.nn.Conv2d(inter_dims[1]+128, inter_dims[2], 3, padding=1)
        if 16 in use_feature:
            inter_dims[1] += dim_add #256
        
        if 32 in use_feature:
            inter_dims[1] += dim_add32 #256
            layer_dim = 32*32
        

        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])

        # 64*4
        hidden_heatmap_dim = self.heatmap_size[0] * 16      
        heatmap_dim = self.heatmap_size[0]* self.heatmap_size[1]

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(layer_dim),
            nn.Linear(layer_dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        )
        final_inchan = inter_dims[2]
        #self.final_layer = nn.Conv1d(
        self.final_layer = nn.Conv2d(
        
            in_channels=final_inchan, 
            out_channels=final_channel,  
            kernel_size=1,#1,
            stride=1, #2 # 1
            padding=0  # if FINAL_CONV_KERNEL = 3 else 1
        )
        
        
        self.check_dim()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, feature: Tensor = None):

        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        
        if 16 in self.use_feature:
            feature16 = self.adapter(feature[0])
            feature16 = _expand(feature16, bbox_mask.shape[1])
            x = torch.cat([x, feature16], dim=1)

        if 32 in self.use_feature:
            feature32 = self.adapter32(feature[1])
            x = x.flatten(2)
            x = self.mlp_head32(x)
            x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=32,p2=32)
            feature32 = _expand(feature32, bbox_mask.shape[1])
            x = torch.cat([x, feature32], dim=1)
                    
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        x = x.flatten(2)
        x = self.mlp_head(x)
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])

        x = self.final_layer(x)
        
        return x


    def check_dim(self):
        print("____check dim in PoseHeadSmallConv____") # src_proj =  batch, src_proj_c, h, w    bbx_mask = batch, num_queries, nheads, h, w
        x = torch.zeros((4, self.dim-8, 16, 16))
        bbox_mask = torch.zeros((4,15,8, 16, 16))
        print("x, bbox_mask", x.shape, bbox_mask.shape)
        
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        print("torch.cat scr_proj + bbox_masks ", x.shape) # batch*num_queries, scr_proj_c+nheads, h, w
        
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        print("lay1 x ", x.shape) # 80 264, 8, 8
    
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        print("lay2 x ", x.shape) # 80, 128, 8, 8

        if 16 in self.use_feature:
            feature16 = torch.zeros((4, 256, 16, 16))
            print("feature = ", feature16.shape)
            feature16 = self.adapter(feature16)
            print("feature16 adapater = ", feature16.shape)
            feature16 = _expand(feature16, bbox_mask.shape[1])
            print("feature16 expand = ", feature16.shape)
            x = torch.cat([x, feature16], dim=1)
            print("x+feautre = ", x.shape)

        if 32 in self.use_feature:
            feature32 = torch.zeros((4, 128, 32, 32))
            print("feature32 = ", feature32.shape)
            feature32 = self.adapter32(feature32)
            print("feature32 adapater32 = ", feature32.shape)
            x = x.flatten(2)
            x = self.mlp_head32(x)
            x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=32,p2=32)
            print("x up sample using mlp_had32 = ", x.shape)
            feature32 = _expand(feature32, bbox_mask.shape[1])
            print("feature32 expand = ", feature32.shape)
            x = torch.cat([x, feature32], dim=1)
            print("x+feautre32 = ", x.shape)


        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        print("lay3 x ", x.shape) 
        
        x = x.flatten(2)
        x = self.mlp_head(x)
        print("mlp head x ", x.shape) 
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        
        x = self.final_layer(x)
        print("final x ", x.shape) 
        #x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        print("reshape x ", x.shape) 



class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        # q : h = batch * num_queries * 256    
        # k : memory = 4 * 256 * 8 * 8
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

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

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1) 
        #print("KL loss = ", loss.type())

    def forward(self, output_x, output_y, target_x, target_y, target_weight):
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_x_gt = target_x[:,idx].squeeze()
            coord_y_gt = target_y[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()
            loss += (self.criterion(coord_x_pred,coord_x_gt).mul(weight).mean()) 
            loss += (self.criterion(coord_y_pred,coord_y_gt).mul(weight).mean())
        return loss / num_joints 


class NMTNORMCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTNORMCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
 
        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
        
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        #if torch.any(gtruth >= 256):
        #    print(gtruth)

        #print(scores.shape, gtruth.shape, gtruth, gtruth.dtype)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.to(labels).type(torch.float)  # or .cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            #print(tmp_.get_device(), tdata.get_device(), labels.get_device())
            #print(tmp_.type(), tdata.type(), labels.type())
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
            loss = torch.mean(self.criterion_(scores, gtruth)) 
        else:
            loss = self.criterion_(scores, gtruth)
        
        #print("loss = ", loss.type())
        return loss

    def forward(self, output_x, output_y, target, target_weight):
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)
        loss = 0
        if batch_size == 0: return 0.

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_gt = target[:,idx].squeeze()
            weight = target_weight[:,idx].squeeze()
            #print(coord_x_pred.shape, coord_y_pred.shape, coord_gt.shape, weight.shape)
            # (b, 256) (b, 2) (b, )
            #print(coord_gt.shape)
            loss += self.criterion(coord_x_pred,coord_gt[:,0]).mul(weight).mean()
            loss += self.criterion(coord_y_pred,coord_gt[:,1]).mul(weight).mean()
        return loss / num_joints

def filter_target_simdr(joints, joints_vis, image_size):
        '''
        :param joints:  [idx_num, num_joints, 2]
        :param joints_vis: [idx_num, num_joints, 1]
        :param image_size: image_size
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        #print(joints.shape)
        bs = joints.shape[0]
        num_joints = joints.shape[1]

        target_weight = torch.ones((bs, num_joints, 1), dtype=torch.float32)#.to(joints)
        #target_weight[:, :, 0] = joints_vis.clone()

        # detect abnormal coords and make the weight 0
        for bs_id in range(bs):
            for joint_id in range(num_joints):
                if joints[bs_id][joint_id][1] < 0:
                    target_weight[bs_id][joint_id] = 0
                    joints[bs_id][joint_id][1]=0
                elif joints[bs_id][joint_id][1] >= image_size:
                    target_weight[bs_id][joint_id] = 0
                    joints[bs_id][joint_id][1] = image_size - 1

                if joints[bs_id][joint_id][0] < 0:
                    target_weight[bs_id][joint_id] = 0
                    joints[bs_id][joint_id][0] = 0
                elif joints[bs_id][joint_id][0] >= image_size:
                    target_weight[bs_id][joint_id] = 0
                    joints[bs_id][joint_id][0] = image_size - 1

        return target_weight,joints  

class PostProcessPoseDR(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs):
        
        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        #print("Post Porcess Pose")
        x_coord = outputs['x_coord']
        y_coord = outputs['y_coord']
        #print(x_coord.shape, y_coord.shape)

        prob_x = F.softmax(x_coord, -1)
        prob_y = F.softmax(y_coord, -1)
        #print(prob_x.shape, prob_y.shape, prob_x[0][0])

        max_val_x, preds_x = prob_x.max(-1, keepdim=True)
        max_val_y, preds_y = prob_y.max(-1, keepdim=True)
        #print(max_val_x.shape, preds_x.shape, max_val_x[0][0], preds_x[0][0])
        
        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        
        pose_output = torch.ones([x_coord.shape[0], x_coord.shape[1], prob_x.shape[2], 3])
        pose_output[:, :, :, 0] = torch.squeeze(preds_x)
        pose_output[:, :, :, 1] = torch.squeeze(preds_y)
        pose_output[:, :, :, 2] = torch.squeeze(max_val_x)

        pose_output = pose_output.cpu().numpy()

        return pose_output

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # max?????? ?????? ????????? 2????????? 1???????????? ??? ??????.
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    # ????????? index, ??? ??????
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    # batch , joint?????? ????????? ????????? array ??????
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # (batch, joints, 2) ????????? ??????. (idx??? ????????????)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # (batch ,joints) ?????? x, y ????????? prediction. ??? ?????? ?????? x, y
    # cv2????????? x,y??? ???????????? ???. ( ????????? ?????? ?????? )
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # np.greater -> ???????????? 0?????? ????????? ????????? False. ?????? True
    # tile ???????????? (batch, joint, 2) ??? preds??? ??????????????? ????????????.
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32) # TF?????? 1, 0 ??????

    # MASK ??????, ??? ???????????? 0????????? ????????? ?????? 0,0?????? ??????. 
    # ?????? ???????????? pred ?????? ????????? ?????? ????????????.
    preds *= pred_mask
    return preds, maxvals

class PostProcessPoseHM(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, targets):
        
        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())
        #print("---------------postprocesshm-------------")
        pose_hm = outputs['pred_hm']

        batch_size = pose_hm.shape[0]
        num_queries = pose_hm.shape[1]
        num_joints =  pose_hm.shape[2]
        pose_hm = pose_hm.flatten(0,1).cpu().numpy()
        preds, maxval = get_max_preds(pose_hm)

        preds = rearrange(preds, '(bs q) j kp -> bs q j kp', bs=batch_size)
        maxval = rearrange(maxval, '(bs q) j v -> bs q j v', bs=batch_size)

        pose_output = np.ones([batch_size, num_queries, num_joints, 3])

        pose_output[:, :, :, 0] = preds[:, :, :, 0] * 8
        pose_output[:, :, :, 1] = preds[:, :, :, 1] * 8
        pose_output[:, :, :, 2] = maxval[:, :, :, 0]

        return pose_output
