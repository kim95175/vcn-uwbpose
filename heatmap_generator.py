import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
#from torchvision import transforms
import torch.nn as nn
import os 
import glob
import numpy as np
import random

import time
import cv2
from einops import rearrange, reduce, repeat
from PIL import Image


def make_heatmaps(image, heatmaps, np_hm=False):
    if np_hm:
        heatmaps = heatmaps * 255
        heatmaps = heatmaps.astype(np.uint8)
        #
    else:
        heatmaps = heatmaps.mul(255)\
                        .clamp(0, 255)\
                        .byte()\
                        .cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid

class HeatmapGenerator():
    def __init__(self, output_res=64, num_joints=13, sigma=-1, multi=False):
        self.output_res = output_res
        self.num_joints = num_joints
        self.multi = multi
        if sigma < 0:
            sigma = self.output_res/64
        print("sigma = ", sigma)
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        if self.multi:
            #num_person = joints.shape[0]
            hms = np.zeros((joints.shape[0], self.num_joints, self.output_res, self.output_res),
                        dtype=np.float32)
            sigma = self.sigma
            #print(joints.shape)
            for idx_p, p in enumerate(joints):
                #print("p", idx_p, p.shape)
                for idx, pt in enumerate(p):
                    #print(pt.shape)
                    if pt[2] > 0.1:
                        x, y = int(pt[0]*self.output_res), int(pt[1]*self.output_res)
                        if x < 0 or y < 0 or \
                        x >= self.output_res or y >= self.output_res:
                            continue

                        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                        c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                        a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                        cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                        aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                        hms[idx_p, idx, aa:bb, cc:dd] = np.maximum(
                            hms[idx_p, idx, aa:bb, cc:dd], self.g[a:b, c:d])
        else:
            hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                        dtype=np.float32)
            sigma = self.sigma
            for p in joints:
                for idx, pt in enumerate(p):
                    if pt[2] > 0.1:
                        x, y = int(pt[0]*self.output_res), int(pt[1]*self.output_res)
                        if x < 0 or y < 0 or \
                        x >= self.output_res or y >= self.output_res:
                            continue

                        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                        c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                        a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                        cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                        aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                        hms[idx, aa:bb, cc:dd] = np.maximum(
                            hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

print("start - data read ")

#data_path = '/data/nlos/save_data_ver4'
data_path = '/data/nlos/save_data_ver6'
data_path_list = glob.glob(data_path + '/*')
data_path_list = sorted(data_path_list)

hm_size = 64

generator = HeatmapGenerator(output_res=hm_size, sigma=1, multi=True)
dir_cnt = -1
#out_dir = [] #list(range(10))#list(range(11, 45))
#out_dir += list(range(30, 43))
#out_dir += list(range(39, 43))
#out_dir = list(range(42))#list(range(41, 43))
in_dir = [41, 42]
for file in data_path_list:
    if os.path.isdir(file) is True:
        dir_cnt += 1
        #if dir_cnt in out_dir:
        if dir_cnt not in in_dir:
            continue
        
        #pose_file_list = glob.glob(file + '/pose_gt/*.npy')
        #pose_file_list = glob.glob(file + '/gt/*.npy')
        #pose_file_list = glob.glob(file + '/HEATMAP_COOR/*.npy')
        #img_file_list = glob.glob(file + '/IMAGE/*.jpg')
        #pose_file_list = glob.glob(file + '/coord/*.npy')
        pose_file_list = glob.glob(file + '/HEATMAP_COOR/*.npy')
        img_file_list = glob.glob(file + '/image/*.jpg')
        

        pose_file_list = sorted(pose_file_list)
        img_file_list = sorted(img_file_list)
        print("\n\ndir_count = ", dir_cnt)
        print('dir(pose):', file, '\t# of data :', len(pose_file_list))
        print('dir(img):', file, '\t# of data :', len(img_file_list))
        if len(img_file_list) != len(pose_file_list) or len(pose_file_list) == 0 :
            continue

        pose0 = pose_file_list[0]
        print(pose0)
        save_dir = file +'/HEATMAP_MULTI64/'.format(hm_size)
        os.makedirs(save_dir, exist_ok=True)
        heatmap_file_list = glob.glob(save_dir + "*.npy")
        print('dir(hm):', file, '\t# of data :', len(heatmap_file_list))
        #if len(heatmap_file_list) == len(pose_file_list):
        #    continue
        img_dir = f'vis/heatmap64'
        os.makedirs(img_dir, exist_ok=True)
        #print(save_dir, img_dir)

        for cnt in range(len(pose_file_list)):
            pose = pose_file_list[cnt]
            name = pose.split('/')[-1]
            save_name = save_dir+name
            np_pose = np.load(pose)
            #print(np_pose.shape)
            np_joints = np.delete(np_pose,(1,2,3,4), axis=1)
            if cnt % 3000 == 1:
                print(np_joints[0][0][0])
            #if print(os.path.isdir(file+'/imgfeature')):
            np_joints[:, :, 0] /= 640#128 #640
            np_joints[:, :, 1] /= 480 #128
            hms = generator(np_joints)
            #print(hms.shape)

            #np.save(save_name, hms)
            if cnt % 64 == 0:
                print(f"[{name}]{pose}({np_pose.shape}) -> {save_name}({hms.shape})")
                #print(np_joints[:, :, 2])
                img = cv2.imread(img_file_list[cnt])
                num = name.split('.')[0]
                if len(hms.shape) > 3:
                    for num_p in range(hms.shape[0]):
                        heatmap_img = make_heatmaps(img, hms[num_p], np_hm=True)
                        cv2.imwrite(f'{img_dir}/{dir_cnt}_{num}[{num_p}].png', heatmap_img)
                        print(f"save_heatmap to {dir_cnt}_{num}[{num_p}].png : img = {img_file_list[cnt]}")
                else:
                    heatmap_img = make_heatmaps(img, hms, np_hm=True)
                    #cv2.imwrite(f'{img_dir}/{dir_cnt}_{num}.png', heatmap_img)
                    print(f"save_heatmap to {dir_cnt}_{num}.png : img = {img_file_list[cnt]}")
    

