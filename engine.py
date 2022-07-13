# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
from operator import is_
import os
import sys
from typing import Iterable
import time
from datetime import datetime
from pathlib import Path
from torchvision import transforms

import torch
import torch.nn.functional as F
import cv2
import numpy as np

import util.misc as utils
from util import box_ops
from evaluate import AP, mAP, mIOU, mkpIOU, class_ap, pose_AP
from visualize import save_batch_heatmaps, visualize_result, vis_featuremap, save_pred_feature, vis_gram


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    is_pose: str = 'simdr', feature_type: str ='x'):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000 if is_pose is not None else 500
    start_time = time.time()

    for samples, targets, cds, hms, features in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        rftr_target = []
        neg_list = []
        
        if is_pose is not None:
            assert len(targets) == len(cds)
        for i in range(len(targets)):
            t = targets[i]
            cd = cds[i]
            hm = hms[i]
            feature = features[i]
            target_class=t[:,-1]
            
            if -1 in target_class:
                if t.shape[0] == 1:
                    neg_list.append(i)
                    continue
                class_exception = target_class != -1
                t = t[class_exception]
                
            rftr_t = {'labels':t[:, -1].long().to(device), 'boxes':t[:, :-1].to(device), 'cd':cd, 'hm':hm, 'features':feature}
            rftr_target.append(rftr_t)

        if len(neg_list) > 0:
            for neg_idx in range(len(neg_list)):
                neg = neg_list[neg_idx] - neg_idx
                samples = torch.cat([samples[0:neg], samples[neg+1:]])
        
        targets= rftr_target
        #print(features[0])

        if feature_type != 'x':
            outputs = model(samples, targets)
        else:
            outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            continue
            #sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("time: ", str(datetime.now()))
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, 
                data_loader, device, output_dir, val_vis=False, 
                is_pose='simdr', img_dir = None, boxThrs=0.5, 
                epoch=-1, dr_size = 512, feature_type: str ='x', soft_nms=False):
    output_folder = img_dir
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    #ap_threshold_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap_threshold_list = [0.5]
    #if epoch == -1:
    #    ap_threshold_list += [0.75]

    ap_metric = []
    if 'bbox' in postprocessors.keys():
        ap_metric += ['bbox']
    
    if 'posedr' in postprocessors.keys() or 'posehm' in postprocessors.keys():
        ap_metric += ['pose']
    

    #ap_metric = ['bbox']
    pose_ap_results = {k:[] for k in ap_threshold_list}
    bbox_ap_results = {k:[] for k in ap_threshold_list}
    
    
    origin_size = torch.tensor([512, 512])
    #origin_size = torch.tensor([256, 256])
    print_freq = 50 if is_pose is not None else 20
    k = 0

    src_count = np.zeros(25)
    for samples, targets, masks, cds, hms, ids, imgs, features in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(ids)
        
        rftr_target = []
        neg_list = []
        for i in range(len(targets)):
            t = targets[i]
            mask = masks[i]
            cd = cds[i]
            hm = hms[i]
            feature = features[i]
            target_class=t[:,-1]
            
            if -1 in target_class:
                if t.shape[0] == 1:
                    neg_list.append(i)  # target이 -1 class 하나뿐인 경우 sample, target에서 제외하고 넘어감
                    continue
                class_exception = target_class != -1  # 아닐 경우 -1인 target만 제거
                targets[i] = targets[i][class_exception]
                
            rftr_t = {'labels':targets[i][:, -1].long().to(device), 'boxes':targets[i][:, :-1].to(device), \
                        'mask_size':mask, 'cd':cd, 'hm':hm, 'features':feature, 'orig_size':origin_size.to(device), \
                            'image_id':ids[i][0], 'f_name':ids[i][1]} 
            rftr_target.append(rftr_t)

    
        if len(neg_list) > 0:
            for neg_idx in range(len(neg_list)):
                neg = neg_list[neg_idx] - neg_idx
                samples = torch.cat([samples[0:neg], samples[neg+1:]])

        targets= rftr_target
        
        if feature_type != 'x':
            outputs = model(samples, targets)
        else: 
            outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if 'bbox' in postprocessors.keys():
            results = postprocessors['bbox'](outputs, orig_target_sizes, nms=soft_nms)
        if 'posedr' in postprocessors.keys():
            res_pose = postprocessors['posedr'](results, outputs)
                
            if val_vis and 'pred_hm' in outputs:
                batch_hm = outputs['pred_hm']
                save_batch_heatmaps(imgs, batch_hm, targets, img_dir, results)
            

        elif 'posehm' in postprocessors.keys():
            res_pose = postprocessors['posehm'](results, outputs, targets)
            #if feature_type != 'x' and val_vis:
            #    vis_featuremap(imgs, targets, outputs, features, img_dir)
                #vis_gram(imgs, targets, outputs, features, img_dir)

            if val_vis:
                batch_hm = outputs['pred_hm']
                save_batch_heatmaps(imgs, batch_hm, targets, img_dir, results)

        k +=1
        for i in ap_threshold_list:
            #if is_pose is not None:
            if 'pose' in ap_metric:
                pose_ap_result = pose_AP(targets, results, res_pose, imgs,  IOUThreshold=i, vis=val_vis, img_dir=output_folder, boxThrs=boxThrs, dr_size=dr_size, pose_method=is_pose)
                pose_ap_results[i].extend(pose_ap_result)

            if 'bbox' in ap_metric:
                bbox_ap_result = AP(targets, results, imgs, IOUThreshold=i, vis=val_vis, img_dir=output_folder, boxThrs=boxThrs)
                bbox_ap_results[i].extend(bbox_ap_result)
            
    print(src_count)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    #print(epoch % 5)
    ap_output_dir = Path(output_dir)
    if ap_output_dir and utils.is_main_process():
        with (ap_output_dir / "mAP.txt").open("a") as f:
            if 'pose' in ap_metric:
                for i in ap_threshold_list:
                    print(f"pose_mAP@{i} : {mAP(pose_ap_results[i])}")
                    if epoch is not -1: f.write(f"epoch {epoch}: pose_mAP@{i} : {mAP(pose_ap_results[i])}\n")
                    if epoch is not -1: f.write(f"epoch {epoch}: pose_mOKS : {mIOU(pose_ap_results[i])}\n")
                
                    print(f"pose_mOKS : {mIOU(pose_ap_results[i])}")
                    print(f"pose_mkpOKS : {mkpIOU(pose_ap_results[i])}")

            if 'bbox' in ap_metric:
                for i in ap_threshold_list:
                    print(f"bbox_mAP@{i} : {mAP(bbox_ap_results[i])}")
                    if epoch is not -1: 
                        if 'pose' in ap_metric and epoch % 10 != 0:
                            continue
                        f.write(f"epoch {epoch}: bbox_mAP@{i} : {mAP(bbox_ap_results[i])}\n")
                    
                    print(f"bbox_mIOU@ : {mIOU(bbox_ap_results[i])}")
                    if epoch is not -1: 
                        if 'pose' in ap_metric and epoch % 10 != 0:
                            continue
                        f.write(f"epoch {epoch}: bbox_mIOU@ : {mIOU(bbox_ap_results[i])}\n")

    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats 
