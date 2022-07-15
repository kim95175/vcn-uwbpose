from re import sub
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
print(torchvision.__path__)
from torchvision.ops import roi_align
#from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os 
import glob
import numpy as np
#import jax.numpy as np
import random

import time
import cv2
from einops import rearrange, reduce, repeat
#from PIL import ImageChop

from util import box_ops
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
#import warnings
#warnings.filterwarnings('ignore')
from collections import deque


import queue
class MovAvgFilter:
    def __init__(self, _n,cutoff):
        self.prevAvg = np.zeros((8, 8, 1024-cutoff))
        # 가장 최근 n개의 값을 저장하는 큐
        self.xBuf = queue.Queue()

        # 초기화로 n개의 값을 0으로 둡니다.
        for _ in range(_n):
            self.xBuf.put(np.zeros((8, 8, 1024-cutoff)))
        # 참조할 데이터의 갯수를 저장합니다.
        self.n = _n
    
    def movAvgFilter(self, x):
        # 큐의 front 값은 x_(k-n) 에 해당합니다.
        front = self.xBuf.get()
        # 이번 스텝에 입력 받은 값을 큐에 넣습니다.
        self.xBuf.put(x)
        
        avg = self.prevAvg + (x - front) / self.n     
        self.prevAvg = avg
        
        return avg
print("start - data read ")

#data_path = '/data/nlos/save_data_ver4'
data_path = '/data/nlos/save_data_ver6'
data_path_list = glob.glob(data_path + '/*')
data_path_list = sorted(data_path_list)
#print(data_path_list)

hm_size = 128

dir_cnt = -1
out_dir = [] 
#out_dir = list(range(41, 43))
target_dir =[]
num_tx = 3
num_rx = 3
cutoff = 0

stack_avg = 8
frame_stack = deque(maxlen=stack_avg)


vis_dir = 'vis/sig_vis8/'
os.makedirs(vis_dir, exist_ok=True)
print(vis_dir+f'imgtest-test-.png')
#show_dir = [0, 38, 41, 42] #, 28, 42] #[18, 28]#[38, 42]
show_dir = list(range(43))
show_dir = [38]
#show_dir = [19, 33]
target_dir = ['imgfeature', 'box_people'] #'HEATMAP128','imgfeature128']
#target_dir = ['mask', 'HEATMAP_COOR']
#target_dir = ['HEATMAP_COOR']#,'HEATMAP_TD']

dropblock_size = 8
for data_dir in data_path_list:
    if os.path.isdir(data_dir) is True:
        dir_cnt += 1
        if dir_cnt not in show_dir:
            continue
        #if dir_cnt in out_dir:
        #    continue
        print(f"[{dir_cnt}] {data_dir}")
        
        sub_data_path_list = glob.glob(data_dir + '/*')
        sub_data_path_list = sorted(sub_data_path_list)

        cnt_list = []
        
        for sub_dir in sub_data_path_list:
            if os.path.isdir(sub_dir) is True:
                #print(sub_dir)
                sub_dir_name = sub_dir.split('/')[-1]
                if sub_dir_name not in target_dir:
                    continue
                file_list = glob.glob(sub_dir + '/*')
                file_list = sorted(file_list)
                
                print(f'dir({sub_dir_name}):', sub_dir, '\t# of data :', len(file_list))
                cnt_list.append(len(file_list))
                #movavg_filter = MovAvgFilter(120, cutoff)
                cnt = 0
                for f_name in file_list:
                    #if cnt >= 10:
                    #    break
                    cnt += 1
                    k = f_name.split('/')[-1]
                    file_num = k.split('.')[0]
                    file_ext = k.split('.')[1]
                    file_num = file_num.zfill(8)
                    new_name = sub_dir + '/' + file_num + '.' + file_ext
                    if sub_dir_name == 'imgfeature':
                        feature = np.load(f_name)
                        #f feature.max() > 6:
                        #    print(f_name, feature.max(), feature.min())
                        #    feature = np.clip(feature, 0.0, 6)
                        #    print(f_name, feature.max(), feature.min())
                            #np.save(f_name, feature)
                    if sub_dir_name =='mask':
                        mask = np.load(f_name)
                        print(mask.shape)
                        if len(mask.shape) != 3:
                            print(mask.shape)
                        break
                    if sub_dir_name =='HEATMAP_TD': #28':
                        hm = np.load(f_name)
                        if hm.shape[0] == 0:
                            print(hm.shape)
                    if sub_dir_name =='box_people':
                        target = np.load(f_name)

                    if sub_dir_name =='HEATMAP_COOR':
                        mask = np.load(f_name)
                        #print(mask.shape)
                        #if len(mask.shape) != 3:
                        if mask.shape[0] < 1:
                            print(mask.shape)
                        #break
                    if sub_dir_name == 'radar':
                        signal = np.load(f_name)
                        signal = signal[:, :, cutoff:]
                        frame_stack.append(signal)
                        '''
                        #mvavg = movavg_filter.movAvgFilter(signal)
                        #signal = signal - mvavg
                        torch_sig = torch.tensor(signal).float()
                        torch_sig = torch.flatten(torch_sig, 0, 1)
                        torch_sig = torch_sig.unsqueeze(0)
                        #print(torch_sig.shape)
  
                        mask = (torch.rand(torch_sig.shape[0], *torch_sig.shape[2:]) < 0.02).float()
                        block_mask = F.max_pool1d(input=mask[:, None, :],
                                    kernel_size=dropblock_size, 
                                    stride=1, #dropblock_size,
                                    padding=dropblock_size // 2)
                        if dropblock_size % 2 == 0: 
                            block_mask = block_mask[:, :, :-1]
                        block_mask = 1 - block_mask.squeeze(1)
                        #print("block_mask", block_mask[:, None, :].shape)  # 1, 1, 768
                        
                        signal = torch_sig * block_mask[:, None, :]
                        signal = signal * block_mask.numel() / block_mask.sum()
                        #
                        signal = signal[0]
                        print(signal[0])
                        signal = signal.reshape((8, 8, -1)).cpu().numpy()
                        '''

                    


                        if sub_dir_name == 'radar':
                            '''
                            print(f"cnt = {dir_cnt}-{file_num}")
                            print(mvavg.shape, np.min(signal), np.max(signal))

                            torch_sig = torch.tensor(signal).float()
                            torch_sig = torch.flatten(torch_sig, 0, 1)
                            rand_idx = torch.rand((64,))
                            #print(rand_idx)
                            rand_idx[rand_idx>0.7] = 1
                            rand_idx[rand_idx<0.7] = 0
                            rand_idx = rand_idx.to(dtype=torch.bool)
                            #print(rand_idx)
                            torch_sig[rand_idx] = torch.zeros((1024,))
                            '''


                            fig = plt.figure(figsize=(15, 9))
                            #title = plt.suptitle('%d.npy' % int(file_num))
                            gs1 = GridSpec(num_tx, num_rx, left=0.03, right=0.97, wspace=0.10)

                            temp = np.zeros(shape=(num_tx, num_rx))
                            temp_signal_list = np.array(temp, dtype=object)
                            signal_list = np.array(temp, dtype=object)

                            for tx in range(num_tx):
                                for rx in range(num_rx):
                                    temp_signal_list[tx][rx] = fig.add_subplot(gs1[tx, rx])
                                    temp_signal_list[tx][rx].axis(ymin=-3, ymax=3)
                                    temp_signal_list[tx][rx].axes.xaxis.set_visible(False)
                                    temp_signal_list[tx][rx].axes.yaxis.set_visible(False)

                                    signal_list[tx][rx], = temp_signal_list[tx][rx].plot(signal[tx][rx],
                                                                                        label='tx:{}, rx:{}'.format(tx + 1, rx + 1))
                                    temp_signal_list[tx][rx].legend(loc='lower right', fontsize=5)
                            
                            plt.savefig(vis_dir + f'{dir_cnt}_{file_num}-sig.png')

                            mean_rf = np.zeros((8, 8, 1024-cutoff))
                            for i in range(stack_avg):
                                raw_rf = frame_stack[i]
                                mean_rf += raw_rf
                            mean_rf /= stack_avg

                            signal -= mean_rf

                            fig = plt.figure(figsize=(15, 9))
                            #title = plt.suptitle('%d.npy' % int(file_num))
                            gs1 = GridSpec(num_tx, num_rx, left=0.03, right=0.97, wspace=0.10)

                            temp = np.zeros(shape=(num_tx, num_rx))
                            temp_signal_list = np.array(temp, dtype=object)
                            signal_list = np.array(temp, dtype=object)

                            for tx in range(num_tx):
                                for rx in range(num_rx):
                                    temp_signal_list[tx][rx] = fig.add_subplot(gs1[tx, rx])
                                    temp_signal_list[tx][rx].axis(ymin=-3, ymax=3)
                                    temp_signal_list[tx][rx].axes.xaxis.set_visible(False)
                                    temp_signal_list[tx][rx].axes.yaxis.set_visible(False)

                                    signal_list[tx][rx], = temp_signal_list[tx][rx].plot(signal[tx][rx],
                                                                                        label='tx:{}, rx:{}'.format(tx + 1, rx + 1))
                                    temp_signal_list[tx][rx].legend(loc='lower right', fontsize=5)
                            
                            plt.savefig(vis_dir + f'{dir_cnt}_{file_num}-sig_avg.png')
                            
                            #torch_sig = torch.tensor(signal).float()
                            #torch_sig = torch.flatten(torch_sig, 0, 1)
                            #for tx in range(8):
                            #    print(torch.max(torch_sig[tx]), torch.min(torch_sig[tx]))

                            #lam = np.random.rand()+ 0.5
                            #torch_sig = torch_sig * lam
                            '''
                            rand_idx = torch.rand((64,))
                            print(rand_idx)
                            rand_idx[rand_idx>0.7] = 1
                            rand_idx[rand_idx<0.7] = 0
                            rand_idx = rand_idx.to(dtype=torch.bool)
                            print(rand_idx)
                            torch_sig[rand_idx] = torch.zeros((1024,))
                            '''
                            #for tx in range(8):
                            #    print(torch.max(torch_sig[tx]), torch.min(torch_sig[tx]))

                        if sub_dir_name == 'image':
                            image = cv2.imread(f_name)
                            cv2.imwrite(vis_dir+f'{dir_cnt}_{file_num}-img.png', image)

                        #print(f_name, new_name)

                    #os.rename(f_name, new_name)
        
                    break



bbox = torch.from_numpy(target[:, :-1])

#print(bbox.shape, bbox, boxes)
feature = torch.from_numpy(feature).type(bbox.dtype)
boxes = box_ops.box_cxcywh_to_xyxy(bbox)

feature =feature.unsqueeze(0).repeat(boxes.shape[0], 1, 1, 1)
#feature =feature.unsqueeze(0).repeat(boxes.shape[0], 1, 1, 1)
print(boxes.shape, feature.shape)
print(list(boxes))
#list_box = 
# convert to roi
device, dtype = boxes.device, boxes.dtype
ids = torch.cat(
    [
        torch.full((1,1), i, dtype=dtype, layout=torch.strided, device=device) 
        for i in range(len(list(boxes)))
    ],
    dim=0,
)
print(ids.shape, boxes.shape)
ftr_size = 16
boxes = torch.mul(boxes, ftr_size)

rois = torch.cat([ids, boxes], dim=1)
#rois = list(boxes.unsqueeze(1))
print(rois) #, rois[0].shape)

#print(feature.dtype, rois.dtype)
#spatial_scale = ftr_size / img_size
roi_result = roi_align(feature, rois, output_size=(4,4), spatial_scale=1.0)
print(roi_result.shape)
'''
for i in range(boxes.shape[0]):
    print( int(boxes[i][1]), int(boxes[i][3]), int(boxes[i][0]), int(boxes[i][2]))

    print(feature[0][0][int(boxes[i][1]):int(boxes[i][3]+1)][int(boxes[i][0]):int(boxes[i][2])+1]  )
    print(roi_result[i][0])
'''

image = torch.arange(0., 49).view(1, 1, 7, 7).repeat(2, 1, 1, 1)
image[0] += 10
print('image: ', image, image.shape)

# for example, we have two bboxes with coords xyxy (first with batch_id=0, second with batch_id=1).
boxes = torch.Tensor([[0, 1, 0, 5, 4],
                     [1, 0.5, 3.5, 4, 7]])
print(boxes.shape)

roi_result = roi_align(image, boxes, output_size=(4,4), spatial_scale=1.0, aligned=True)
print(roi_result)