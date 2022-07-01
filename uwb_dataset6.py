import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
#from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os 
import glob
import numpy as np
import random
import queue

import time
import cv2
from scipy import signal
from einops import rearrange, reduce, repeat
from PIL import Image

from collections import deque


class UWBDataset(Dataset):
    def __init__(self, args, mode='train'):
        
        '''
        dataset 처리
        rf와 이미지의 경우에는 init 할 때부터 읽어와서 메모리에 올리지만 gt는 데이터를 활용할 때마다 load함.
        mode - train : 학습을 위함.  rf, gt, img 다 있는 경우
                test : test를 위함. rf, gt, img 다 있는 경우 

        '''
        
        self.mode = mode


        self.load_cd = True if args.pose is not None else False
        if mode =='test':
            self.load_cd = True
        self.load_hm = True if (args.pose == 'hm' or args.pose =='hmdr') else False
        self.load_mask = True if args.pose is not None and mode != 'train' else False
        self.load_img = args.vis and mode != 'train'
        self.load_feature = args.feature is not 'x'
        self.feature = args.feature

        if args.feature_train:
            self.load_cd, self.load_mask, self.load_img, self.load_hm = False, False, False, False
            self.load_feature = True

        self.is_normalize = False #True
        self.is_ftr_normalize = False #True
        self.cutoff = args.cutoff
        
        self.print_once = True

        self.frame_stack_num = args.stack_num
        self.frame_skip = args.frame_skip
        
        self.mixup_prob = args.mixup_prob
        self.mixup_alpha = 1.5

        self.model_debug = args.model_debug
        self.erase_size = 8

        self.three = args.three
        self.num_txrx = args.num_txrx

        if self.load_feature:
            self.mixup_prob = 0.
        
        self.stack_avg = args.stack_avg #True if args.stack_num == args.frame_skip else False #True
        assert self.frame_stack_num <= args.stack_avg 
        if self.stack_avg > 0:
            outlier_by_ewma = self.stack_avg + 10

        
        data_path = '/data/nlos/save_data_ver6'
        data_path_list = glob.glob(data_path + '/*')
        data_path_list = sorted(data_path_list)
        #print(data_path_list)
        rf_data = []  # rf data list
        raw_list = []
        target_list = []

        mask_list = []  # ground truth mask
        hm_list = []
        cd_list = []

        img_list = []
        filename_list =[]
        
        three_dir = [17, 21, 22, 25, 26, 31, 32]
        three_list = []

        feature_list = []
        outlier_list = []
        
        remove_dir = []
        print("start - data read ", mode)
        # 데이터셋 세부 내용은 비가시drive/미팅자료/8월 데이터 수집 참고
        
        test_dir = args.test_dir
        
        # 
        # 9 - B_one_1 3000
        # 6 - A_two_1 2100
        # 38 - test_C_four 2500
        # 26 - D_four_2 3000
        # 35 - D_two_3 2000
        # 37 - cloth 39 - 스티로폼 40 - Wall
        # 41 - E, 42 - F
        #valid_dir = [8, 17] #list(range(18, 30)) #[8, 17] 
        valid_dir = [9, 6, 38, 37, 39, 40, 41, 42]
        valid_dir += [0]
        if args.eval:
            valid_dir = list(range(1, 30))
                   
        train_dir = [x for x in list(range(43)) if x not in valid_dir] # 29
        
        if self.model_debug or args.eval:
            train_dir = [6]
        

 
        train_outlier_idx = []
        test_outlier_idx = []
        #test_outlier_idx = [0, 8835, 8835+12000, 8835+12000+13000]
        if mode == 'train':
            print("train_dir = ", train_dir)
            print("valid_dir = ", valid_dir)
            print("remove_dir", remove_dir)
        if mode == 'test':
            if self.model_debug:
                test_dir = [6]#[38] # 18, 22
            elif not args.eval:
                test_dir = [42]

            test_set = args.test_dir[0]
            if test_set == 51:
                test_dir = [9]
            elif test_set == 52:
                test_dir = [6] 
            elif test_set == 53:
                test_dir = [38]
                outlier_list = list(range(2000, 4000))
            elif test_set == 54:
                test_dir = [38]
                outlier_list = list(range(2000))
            elif test_set == 55:
                test_dir = [9, 6, 38]
            
            elif test_set == 61:
                test_dir = [42]
                outlier_list = list(range(1000,3001))
            elif test_set == 62:
                test_dir = [42]
                outlier_list = list(range(1000))
                outlier_list += list(range(2000,3001))
            elif test_set == 63:
                test_dir = [42]
                outlier_list = list(range(2000))
                outlier_list += list(range(2600,3001))
            elif test_set == 64:
                test_dir = [42]
                outlier_list += list(range(2600))
            elif test_set == 65:
                test_dir = [42]

            elif test_set == 66:
                test_dir = [41]
                outlier_list = []
                outlier_list += list(range(200, 500-self.stack_avg))
                outlier_list += list(range(600, 800-self.stack_avg))
                outlier_list += list(range(1000,3001))
                #outlier_list = list(range(600-self.stack_avg-10))
                #outlier_list += list(range(800,3001))
            elif test_set == 67:
                test_dir = [41]
                outlier_list = list(range(1000))
                outlier_list += list(range(2000,3001))
            elif test_set == 68:
                test_dir = [41]
                outlier_list = list(range(2000))
                outlier_list += list(range(2600,3001))
            elif test_set == 69:
                test_dir = [41]
                outlier_list = list(range(2600))
            elif test_set == 70:
                test_dir = [41]
                outlier_list = []
                outlier_list += list(range(200, 500-self.stack_avg))
                outlier_list += list(range(600, 800-self.stack_avg))
            
            elif test_set == 71:
                test_dir = [37]
            elif test_set == 72:
                test_dir = [39]
            elif test_set == 73:
                test_dir = [40]
                
            print("test_dir = ", test_dir)
        
        print("outlier_list = ", len(outlier_list))

        dir_count = 0  # dicrectory index

        rf_index = -1  # rf data index
        target_index = -1
        rf_frame = -1 # index that saved in raw_list

        mask_index = -1
        hm_index = -1
        cd_index = -1

        img_index = -1
        filename_index = -1

        feature_index = -1

        #frame_stack = deque(maxlen=self.frame_stack_num)
        frame_stack = deque(maxlen=self.stack_avg)
        not_stacked_list = []
        for file in data_path_list:
            if dir_count in remove_dir:
                dir_count += 1
                continue

            if mode == 'train' and dir_count not in train_dir:
                dir_count += 1
                continue
            elif mode == 'test' and dir_count not in test_dir:
                dir_count += 1
                continue

            if os.path.isdir(file) is True:
                # 각 폴더 안의 npy 데이터
                rf_file_list = glob.glob(file + '/radar/*.npy')
                rf_file_list = sorted(rf_file_list)
                print('\n\n\tdir_count:', dir_count,'dir(raw):', file)
                print('\t# of data :', len(rf_file_list), "rf_idx =", rf_index)
                if dir_count in three_dir:
                    print("\t************")
                #print(rf_file_list)
                frame_stack.clear()
                dir_rf_index = -1

                for rf in rf_file_list:
                    rf_index += 1
                    if dir_count in three_dir:
                        three_list.append(rf_index)
                        
                    if rf_index in outlier_list:
                        if mode=='train' and rf_index in train_outlier_idx:
                            print("train_outlier_idx = ", rf_index, rf)
                        
                        if mode=='test' and rf_index in test_outlier_idx:
                            print("test_outlier_idx = ", rf_index, rf)
                        dir_rf_index = -1
                        frame_stack.clear()
                        continue
                   
                    dir_rf_index += 1
                    raw_rf_load = np.load(rf)
                    
                        
                    if dir_count in [41, 42]:
                        #print(raw_rf_load.shape)
                        #print(np.max(raw_rf_load[6]), np.max(raw_rf_load[7]))
                        raw_rf_load[(6, 7), :, :] = raw_rf_load[(7, 6), :, :]
                        raw_rf_load[:, (6, 7), :] = raw_rf_load[:, (7, 6), :]
                        #print(np.max(raw_rf_load[6]), np.max(raw_rf_load[7]))

                    if self.num_txrx == 4:
                        raw_rf_load = raw_rf_load[(1, 2, 5, 6), :, :]
                        raw_rf_load = raw_rf_load[:, (1,2,5,6), :]
                    elif self.num_txrx == 2: 
                        raw_rf_load = raw_rf_load[(1, 6), :, :]
                        raw_rf_load = raw_rf_load[:, (1, 6), :]

                    temp_raw_rf = raw_rf_load[:, :, self.cutoff:]
                    #temp_raw_rf = raw_rf_load[:, :, :-self.cutoff].copy()

                    ##----- normalization after ------
                    if self.is_normalize is True:
                        for i in range(temp_raw_rf.shape[0]):
                            for j in range(temp_raw_rf.shape[1]):
                                stdev = np.std(temp_raw_rf[i, j])
                                #temp_raw_rf[i,j] = temp_raw_rf[i,j] - np.mean(temp_raw_rf[i,j])
                                temp_raw_rf[i, j] = temp_raw_rf[i, j]/stdev
                        #raw_rf_load = raw_rf_load - np.mean(raw_rf_load)

                    if dir_rf_index == 1000:
                        print("dir_rf_index {} max, min, mean(raw) = ".format(dir_rf_index), np.max(raw_rf_load), np.min(raw_rf_load), np.mean(raw_rf_load))
                        print("dir_rf_index {} max, min, mean(input) = ".format(dir_rf_index), np.max(temp_raw_rf), np.min(temp_raw_rf), np.mean(temp_raw_rf), raw_rf_load.shape)
                                   
                    
                    temp_raw_rf = torch.tensor(temp_raw_rf).float()
                    temp_raw_rf = torch.flatten(temp_raw_rf, 0, 1)
                    #temp_raw_rf = rearrange(temp_raw_rf, 'tx rx len -> (tx rx) len')
                    raw_list.append(temp_raw_rf)
                    rf_frame +=1
                    frame_stack.append(rf_frame)

                    #if len(frame_stack) == self.frame_stack_num and dir_rf_index >= outlier_by_ewma:
                    if len(frame_stack) == self.stack_avg and dir_rf_index >= outlier_by_ewma:
                        rf_data.append(list(frame_stack))
                    else:
                        not_stacked_list.append(rf_index)
                                        
                    if rf_index %10000 == 1000 and len(frame_stack) == self.stack_avg:
                        print(f"rf_index {rf_index}, rf_frame {rf_frame}, rf_file = {rf}, frame_stack ={frame_stack[0]}~{frame_stack[-1]}[{self.stack_avg}]")
                        if self.frame_stack_num > 1 or self.stack_avg > 1:
                            tmp_rf = []
                            stack_idx = []
                            mean_rf = torch.zeros((self.num_txrx*self.num_txrx, 1024-self.cutoff))
                            for i in range(self.stack_avg):
                                if self.stack_avg > 1:
                                    raw_rf = raw_list[frame_stack[i]]
                                    mean_rf += raw_rf
                                    if i >= self.stack_avg - self.frame_stack_num and i%self.frame_skip == self.frame_skip-1:
                                        tmp_rf.append(raw_rf)
                                        stack_idx.append(i)
                            #tmp_rf.append(raw_list[frame_stack[-1]])
                            print("stack idx = ", stack_idx)
                            #rf = torch.stack(tuple(tmp_rf), 0)
                            rf = torch.stack(tmp_rf, 0)
                            mean_rf /= self.stack_avg
                            mean_rf = mean_rf[None, :, :]
                            mean_rf = mean_rf.repeat(rf.shape[0], 1, 1)
                            print("mean ", mean_rf.shape, mean_rf[0,0,:3], torch.min(mean_rf), torch.max(mean_rf))
                            print("rf ", rf.shape, rf[0,0,:3], torch.min(rf), torch.max(rf))
                            rf -= mean_rf
                            print("rf ", rf.shape, rf[0,0,:3], torch.min(rf), torch.max(rf))

            
                
                if self.load_img:
                    img_file_list = glob.glob(file + '/image/*.jpg')
                    img_file_list = sorted(img_file_list)
                    print('\n\tdir(img):', file, '\t# of data :', len(img_file_list))

                    for img in img_file_list:
                        img_index += 1
                        filename_index += 1
                        if img_index in outlier_list or img_index in not_stacked_list:
                            continue
                        #temp_img = cv2.imread(img)
                        #img_list.append(temp_img)
                        f_name = '{}/pred_feature/{}.npy'.format(file, img.split('/')[-1].split('.')[0])
                        #print(f_name)
                        
                        filename_list.append(f_name)
                        img_list.append(img)

                        if img_index %10000 == 1000:
                            print(f"img_index {img_index} img_shape {cv2.imread(img).shape}")

                if self.load_mask:
                    mask_file_list = glob.glob(file + '/mask/*.npy')
                    #mask_file_list = glob.glob(file + '/HEATMAP_COOR/*.npy')
                    mask_file_list = sorted(mask_file_list)
                    print('\n\tdir(mask):', file, '\t# of data :', len(mask_file_list))
                    assert len(mask_file_list) == len(rf_file_list)
                    for mask in mask_file_list:
                        mask_index += 1
                        if mask_index in outlier_list or mask_index in not_stacked_list:
                            continue
                        
                        mask_np = np.load(mask)
                        if mask_np.shape[0] != 0:
                            #mask_test_np = mask_np.copy()
                            mask_np = mask_np.transpose((1, 2, 0))
                            #mask_np = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
                            mask_np = np.where(mask_np > 0.5, 1.0, 0)
                            if len(mask_np.shape) == 2:
                                mask_np = np.expand_dims(mask_np, axis=0)
                            else:
                                mask_np = mask_np.transpose((2, 0, 1))  
                            mask_size = mask_np.shape[1]*mask_np.shape[2] 
                            mask_np = mask_np.reshape(mask_np.shape[0], mask_size)

                            #mask_test_size = mask_test_np.shape[1]*mask_test_np.shape[2] 
                            #mask_test_np = mask_test_np.reshape(mask_test_np.shape[0], mask_test_size)
                            mask_s = np.sum(mask_np, axis=1) * 18.2
                            
                        else:
                            mask_s = np.ones((1))

                        if mask_index %10000 == 1000:
                            print("mask_shape ", mask, mask_np.shape)
                            print(mask_s)

                        mask_list.append(mask_s)
                
                if self.load_hm:
                    #hm_file_list = glob.glob(file + '/HEATMAP128/*.npy')
                    #hm_file_list = glob.glob(file + '/HEATMAP_MULTI/*.npy')
                    hm_file_list = glob.glob(file + '/HEATMAP_MULTI64/*.npy')
                    
                    #hm_file_list = glob.glob(file + '/pose_gt/*.npy')
                    hm_file_list = sorted(hm_file_list)
                    print('\n\tdir(posehm):', file, '\t# of data :', len(hm_file_list))
                
                    for hm in hm_file_list:
                        hm_index += 1
                        if hm_index in outlier_list or hm_index in not_stacked_list:
                            continue
                        
                        #if True:
                        if hm_index %10000 == 1000:
                            np_hm = np.load(hm)
                            #np_hm = rearrange(np_hm, 'n key h w -> n key (h w)')
                            print("hm_shape ", hm, np_hm.shape)
                                    
                        hm_list.append(hm)
                

                if self.load_cd:
                    #cd_file_list = glob.glob(file + '/coord/*.npy')
                    cd_file_list = glob.glob(file + '/HEATMAP_COOR/*.npy')
                    cd_file_list = sorted(cd_file_list)
                    print('\n\tdir(pose_cd):', file, '\t# of data :', len(cd_file_list))
                
                    for cd in cd_file_list:
                        cd_index += 1
                        if cd_index in outlier_list or cd_index in not_stacked_list:
                            continue
                        np_cd = np.load(cd)
                        np_cd = np.delete(np_cd,(1,2,3,4), axis=1)
                        np_cd[:, :, 0] /= 640#704
                        np_cd[:, :, 1] /= 480#512
                                                        
                        cd_list.append(np_cd)
                        #cd_list.append(visible_nodes)

                        if cd_index %10000 == 1000:
                            print("cd_shape ", cd, np_cd.shape, np_cd[0][0]) #, visible_nodes[0])
                                    
                        #cd_list.append(np_cd)

                target_file_list = glob.glob(file + '/box_people/*.npy')
                target_file_list = sorted(target_file_list)
                print('\n\tdir(target):', file + '/box_people/*.npy')
                print('\t# of data :', len(target_file_list))
                if len(target_file_list) == 0:
                    for _ in range(len(rf_file_list)):
                        target_index += 1
                        if target_index in outlier_list or target_index in not_stacked_list:
                            continue
                        target_list.append(None)
                else:
                    for target in target_file_list:
                        target_index += 1
                        #print(target_index, target[:-4], int(target[-9:-4]))
                        if target_index in outlier_list or target_index in not_stacked_list:
                            continue
                        if target_index % 10000 == 1000:
                            print("target_shape ", target_index, target, np.load(target).shape, np.load(target))
                        target_list.append(np.load(target))
                        #target_list.append(target)

                
                if self.load_feature is True:
                    #feature_file_list = glob.glob(file + '/featuremap/*.npy')
                    if args.feature =='32':
                        feature_file_list = glob.glob(file + '/imgfeature3/*.npy')
                    elif args.feature =='128':
                        feature_file_list = glob.glob(file + '/imgfeature128/*.npy') 
                    else:    #if args.feature == '16':
                        feature_file_list = glob.glob(file + '/imgfeature/*.npy')
                    feature_file_list = sorted(feature_file_list)
                    print('\n\tdir(feature):', file, '\t# of data :', len(feature_file_list))
                
                    for feature in feature_file_list:
                        feature_index += 1
                        if feature_index in outlier_list or feature_index in not_stacked_list:
                            continue
                        
                        #print("mask_shape ", mask.shape)
                        if feature_index %10000 == 1000:
                            np_feature = np.load(feature)#cv2.resize(np.load(feature), (64, 64), interpolation=cv2.INTER_AREA)
                            #np_feature = np_feature[:, ::2, ::2]
                            
                            print("featuremap_shape ", feature, np_feature.shape)
                            print(np_feature.max(), np_feature.min())
                            #feature = feature / 6.0 #
                            if self.is_ftr_normalize:
                                np_feature = np_feature / np.max(np_feature)
                                print(np_feature.max(), np_feature.min())
                        feature_list.append(feature)

            dir_count += 1

        
        #print("not_stacked_list {}: {} ~ {}".format(len(not_stacked_list), not_stacked_list[0], not_stacked_list[-1]))
        self.rf_data = rf_data
        self.mask_list = mask_list
        self.hm_list = hm_list
        self.cd_list = cd_list
        self.img_list = img_list
        self.raw_list = raw_list
        self.feature_list = feature_list
        self.filename_list = filename_list

        self.target_list = target_list

        self.three_list = three_list
        
        #self.human_index = human_index
        print(f"rf\t{len(rf_data)}/{outlier_by_ewma}\t raw\t{len(raw_list)}/{rf_frame}\t target\t{len(target_list)}")
        print(f"pose_cd\t{len(cd_list)}\t pose_hm\t{len(hm_list)}\t mask\t{len(mask_list)}")
        print(f"img\t{len(img_list)}\t file_name\t{len(filename_list)}")
        print(f"feature\t{len(feature_list)}\t 3~4 list\t{len(three_list)}" )

        print(f"rf_data[-1] = {rf_data[-1][0]} ~ {rf_data[-1][-1]}")
        print(f"{target_list[-1]}")
        #print(rf_data)
        if self.mode =='train':
            #assert len(rf_data) == len(hm_list)
            if self.load_cd:
                assert len(rf_data) == len(cd_list)
            assert len(rf_data) == len(target_list)
        assert len(raw_list) == rf_frame + 1

        
        print("end - data read")
        print("size of dataset", len(self.rf_data))

    def __len__(self):    
        return len(self.rf_data)

    def __getitem__(self, idx):
        
        if random.random() < self.three and self.mode == 'train' and len(self.three_list)>0:
            idx = self.three_list[idx%len(self.three_list)]

        mask = np.ones((1))
        img = None
        f_name = None
        cd = None
        hm = None
        feature = np.ones((256, 16, 16))

        rf = self.get_rf(idx)
        #target = np.load(self.target_list[idx])
        target = self.target_list[idx]
        
        if self.load_mask:
            mask = self.mask_list[idx]
    
        if self.load_cd:
            cd = self.cd_list[idx]
        
        if self.load_hm:
            hm = self.get_hm(idx)
        
        if self.load_img:
            img = self.img_list[idx]
            img = cv2.imread(img)
            f_name = self.filename_list[idx]


        if self.load_feature:
            feature_name = self.feature_list[idx]
            feature = np.load(feature_name)
            if self.is_ftr_normalize:
                feature = feature / np.max(feature)
            #feature = feature[:, ::2, ::2] # 32 64 64
            
        if self.mixup_prob > 0.0 and self.mode == 'train' and not self.load_hm:
            if random.random() < self.mixup_prob:
                mixup_idx = random.randint(0, len(self.rf_data)-1)
                if idx != mixup_idx:
                    mixup_rf = self.get_rf(mixup_idx)
                    mixup_target = self.target_list[mixup_idx]
                    #mixup_target = np.load(mixup_target)

                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) #0.5
                    
                    rf = lam * rf + (1.-lam) * mixup_rf
                    target = np.concatenate((target, mixup_target), axis=0) if idx != mixup_idx else target
                    
                    if self.load_cd:
                        mixup_cd = self.cd_list[mixup_idx]
                        cd = np.concatenate((cd, mixup_cd), axis=0) if idx != mixup_idx else cd
                    
                    if self.load_hm:
                        mixup_hm = self.get_hm(mixup_idx)
                        hm = np.concatenate((hm, mixup_hm), axis=0) if idx != mixup_idx else hm

                    if self.load_feature:
                        mixup_feature = np.load(self.feature_list[idx]) if idx != mixup_idx else feature
                        feature = lam * feature + (1.-lam) * mixup_feature
            
            elif random.random() < 0.5 and self.mode == 'train':
                lam = np.random.beta(0.5, 0.5) #np.random.rand() #+ 0.5#np.random.beta(0.5, 0.5)
                rf = lam * rf
                
            '''
            if random.random() < 0.5 and self.mode == 'train':
                rand_idx = torch.rand((self.frame_stack_num, 64))
                rand_idx[rand_idx>0.8] = 1
                rand_idx[rand_idx<0.8] = 0
                rand_idx = rand_idx.to(dtype=torch.bool)
                rf[rand_idx] = torch.zeros((1024-self.cutoff,))

            if random.random() < 0.5 and self.mode == 'train':
            #if False:
                #print(rf.shape)
                mask = (torch.rand(rf.shape[0], *rf.shape[2:]) < 0.02).float()
                block_mask = F.max_pool1d(input=mask[:, None, :],
                            kernel_size=self.erase_size, 
                            stride=1,
                            padding=self.erase_size // 2)
                if self.erase_size % 2 == 0: 
                    block_mask = block_mask[:, :, :-1]
                block_mask = 1 - block_mask.squeeze(1)
                #print("block_mask", block_mask[:, None, :].shape)
                #print(block_mask.shape, block_mask[0])
                #print(block_mask.numel(), block_mask.sum())
                rf = rf * block_mask[:, None, :]
                #rf = rf * block_mask.numel() / block_mask.sum()
            '''
                
        
        if self.mode=='train':
            return rf, target, cd, hm, feature

        else:
            return rf, target, mask, cd, hm, idx, img, feature, f_name
            
       
    def get_rf(self, idx):
        rf = self.rf_data[idx]
        if self.stack_avg > 1:
            tmp_rf = []
            mean_rf = torch.zeros((self.num_txrx*self.num_txrx, 1024-self.cutoff))
            for i in range(self.stack_avg):
                raw_rf = self.raw_list[rf[i]]
                mean_rf += raw_rf
                if i >= self.stack_avg - self.frame_stack_num and i%self.frame_skip == self.frame_skip-1:
                    #print(i, raw_rf.shape, raw_rf[0,:50])
                    tmp_rf.append(raw_rf)
            rf = torch.stack(tuple(tmp_rf), 0)
            mean_rf /= self.stack_avg
            mean_rf = mean_rf[None, :, :]
            mean_rf = mean_rf.repeat(rf.shape[0], 1, 1)
            #print("mean ", mean_rf.shape, mean_rf[0,0,:5], torch.min(mean_rf), torch.max(mean_rf))
            #print("rf ", rf.shape, rf[0,0,:5], torch.min(rf), torch.max(rf))
            rf -= mean_rf
            #print("rf ", rf.shape, rf[0,0,:5], torch.min(rf), torch.max(rf))
                
        else:
            rf = self.raw_list[rf[-1]].unsqueeze(0)
            

        return rf

    def get_hm(self, idx):
        pose = self.hm_list[idx]
        pose = np.load(pose)        
        #pose = np.delete(pose,(1,2,3,4), axis=0)
        return pose


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    rfs = []
    targets = []
    cds = []
    hms = []
    features = []

    for sample in batch:
        #rf = torch.FloatTensor(sample[0]).clone()
        rf = sample[0].clone()
        target = torch.FloatTensor(sample[1]).clone()
        cd = torch.FloatTensor(sample[2]).clone() if sample[2] is not None else None
        hm = torch.FloatTensor(sample[3]).clone() if sample[3] is not None else None
        feature = torch.FloatTensor(sample[4]).clone() if sample[4] is not None else None

        rfs.append(rf)
        targets.append(target)
        cds.append(cd)
        hms.append(hm)
        features.append(feature)

    rfs = torch.stack(rfs)
    features = torch.stack(features)
 
    return rfs, targets, cds, hms, features

def detection_collate_val(batch):
    rfs = []
    targets = []
    masks = []
    cds =[]
    hms = []
    ids = []
    imgs = []
    features = []

    for sample in batch:
        #rf = torch.FloatTensor(sample[0]).clone()  
        rf = sample[0].clone()
        target = torch.FloatTensor(sample[1]).clone()
        mask = sample[2]
        cd = torch.FloatTensor(sample[3]).clone() if sample[3] is not None else None
        hm = torch.FloatTensor(sample[4]).clone() if sample[4] is not None else None
        idx = (sample[5], sample[8])
        img = sample[6]
        feature = torch.FloatTensor(sample[7]).clone() if sample[7] is not None else sample[7]

        rfs.append(rf)
        targets.append(target)
        ids.append(idx)
        masks.append(mask)
        cds.append(cd)
        hms.append(hm)
        imgs.append(img)
        features.append(feature)

    rfs = torch.stack(rfs)

    return rfs, targets, masks, cds, hms, ids, imgs, features


