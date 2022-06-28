import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
import torch
from PIL import Image
from util.misc import nested_tensor_from_tensor_list
from util import box_ops
from models.rftr_pose import get_max_preds
from visualize import imshow_keypoints
from einops import rearrange, repeat

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

COCO_CLASSES = ('person')

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])


skeleton = [[11, 9], [9, 7], [12, 10], [10, 8], 
            [7, 8], [1, 7], [2, 8], 
            [1, 2], [1, 3], [2, 4], [3, 5], [4,6]]

pose_link_color = palette[[
    0, 0, 0, 0,
    7, 7, 7, 
    9, 9, 9, 9, 9
]]

pose_kpt_color = palette[[
    16, 9, 9, 9, 9, 
    9, 9, 0, 0, 0, 
    0, 0, 0
]]

def oks_iou(gt, dt, area, gt_box):
    # nose, //eyes1, eyes2, ears1, ears2,// shoulder1, shoulder2, elbows1, elbows2, wrists1, wrists2, hips1, hips2, knees1, knees2, ankles1, ankles2 
    # 0: nose,  1,2: shoulder1, 3,4 : elbows1, 5,6: wrists1,  7,8: hips1,  9,10: knees1, 11, 12: ankles1
    sigmas = np.array([.26, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    vars = (sigmas * 2) ** 2
    k = len(sigmas)

    xg = gt[:, 0]; yg = gt[:, 1]; vg = gt[:,2]
    xd = dt[:, 0]; yd = dt[:, 1]
    k1 = np.count_nonzero(vg > 0.5)
    
    bb = gt_box
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    x0 = bb[0] - w; x1 = bb[0] + w*2
    y0 = bb[1] - h; y1 = bb[1] + h*2
    if k1 > 0:
        dx = xd - xg
        dy = yd - yg
    else:
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

    #print(area, area + np.spacing(1))
    e = (dx ** 2 + dy ** 2) / vars / (area + np.spacing(1)) / 2
    
    keypoint_ious = np.exp(-e)
    if k1 > 0 : e=e[vg>0.5]

    ious = np.sum(np.exp(-e)) / e.shape[0]
    #print(ious)
    return ious, keypoint_ious


def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
        
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    
    # intersection over union
    result = interArea / union
    assert result >= 0
    return result


def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]
    #print(mrec, mpre)
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]


def ElevenPointInterpolatedAP(rec, prec):

    mrec = [e for e in rec]
    mpre = [e for e in prec]

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]


# Cacluate Average Precision (AP) and visualize results
def AP(targets, results, imgs=None, 
        IOUThreshold = 0.5, method = 'AP', vis=False, 
        img_dir=None, boxThrs=0.5):
    
    #print("==============calcuate ap ==============")
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    #print(targets, results)
    detections, groundtruths, classes = [], [], []
    batch_size = len(targets)
    result = []
    # ground truth data
    for i in range(batch_size):
        # Ground Truth Data
        img_size = targets[i]['orig_size']
        num_obj = targets[i]['labels'].shape[0]
        img_id = targets[i]['image_id']


        bbox = targets[i]['boxes']
        boxes = box_ops.box_cxcywh_to_xyxy(bbox)

        img_h, img_w = img_size
        boxes = torch.mul(boxes, img_h)

        if vis and imgs is not None:
            img = imgs[i]
            ori_size = img_size.int().cpu().numpy()
            gt_img = cv2.resize(img, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_AREA)
            pred_img = gt_img.copy()

        for j in range(num_obj):
            label, conf = targets[i]['labels'][j].item(), 1
            x1, y1, x2, y2 = boxes[j]
            #x1, y1, x2, y2 = box_ops.sanitize_all_coordinates(targets[i]['boxes'][j][0], targets[i]['boxes'][j][1], targets[i]['boxes'][j][2], targets[i]['boxes'][j][3], img_size[0])
            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            groundtruths.append(box_info)
            if label not in classes:
                classes.append(label)
            if vis:
                x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                rand_num = (img_id*x1)%19
                color= COLORS[rand_num]
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 1)
                label_name = COCO_CLASSES[label]
                text_str = '%s: %.2f' % (label_name, conf)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale, font_thickness = 0.6, 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(gt_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)  # draw bbox
                cv2.putText(gt_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA) # append label and confidence score
                cv2.putText(gt_img, 'GT', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
        # Prediction Data
        pred = results[i]
        pred_score = pred['scores']
        pred_label = pred['labels']
        pred_boxes = pred['boxes']
       
        for q in range(pred_boxes.shape[0]):
            label, conf = pred_label[q].item(), pred_score[q].item()
            x1, y1, x2, y2 = pred_boxes[q]
            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            #if pred_score[q] > boxThrs:
            if pred_score[q] > boxThrs:
                detections.append(box_info)
            if label not in classes:
                classes.append(label)
            if vis:
                if pred_score[q] > boxThrs:
                    x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                    rand_num = (img_id*x1)%19
                    color= COLORS[rand_num]
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 1)
                    label_name = COCO_CLASSES[label]
                    text_str = '%s: %.2f' % (label_name, conf)
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale, font_thickness = 0.6, 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(pred_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(pred_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(pred_img, 'pred', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        if vis and i == 0:
            res = np.concatenate((pred_img, gt_img), axis=1)
            #cv2.imwrite(os.path.join(img_dir, str()) +'_{}_bbox.png'.format(img_id), res)

        #if not vis and i >=batch_size/10:
        #    break 

    for c in classes:
        dects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]

        npos = len(gts)

        dects = sorted(dects, key = lambda conf : conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts)
        
    
        avg_iou = 0
        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in range(len(dects)):
            gt = [gt for gt in gts if gt[0] == dects[d][0]]
            iouMax = 0

            for j in range(len(gt)):
                iou1 = iou(dects[d][3], gt[j][3])
                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j

            avg_iou += iouMax

            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        #print(acc_TP, acc_FP)
        rec = acc_TP / npos 
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        det_iou = avg_iou / len(dects) if len(dects) != 0 else 0.
        r = {
            'class' : c,
            'precision' : prec,
            'recall' : rec,
            'iou' : det_iou,
            'AP' : ap,
            'interpolated precision' : mpre,
            'interpolated recall' : mrec,
            'total positives' : npos,
            'total TP' : np.sum(TP),
            'total FP' : np.sum(FP)
        }

        result.append(r)

    return result

# Cacluate Average Precision (AP) and visualize results
def pose_AP(targets, results, res_pose, imgs=None,  
        IOUThreshold = 0.5, method = 'AP', vis=False, 
        img_dir=None, boxThrs=0.5, dr_size=256, pose_method='simdr'):
    
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
    #print(targets, results)
    detections, groundtruths, classes = [], [], []
    batch_size = len(targets)
    result = []

    print_hm = True
    
    # ground truth data
    for i in range(batch_size):
        # Ground Truth Data
        img_size = targets[i]['orig_size']
        ori_size = img_size.int().cpu().numpy()
        num_obj = targets[i]['labels'].shape[0]
        img_id = targets[i]['image_id']
            
        bbox = targets[i]['boxes']
        boxes = box_ops.box_cxcywh_to_xyxy(bbox)
        mask_size = targets[i]['mask_size']
        '''
        if pose_method == 'hm':
            poses = targets[i]['hm']
            pose_hm = poses.cpu().numpy()
            #print(pose_hm.shape)
            num_people = pose_hm.shape[0]
            num_joints =  pose_hm.shape[1]
            preds, maxval = get_max_preds(pose_hm)
            gt_poses = np.ones([num_people, num_joints, 3])

            gt_poses[:, :, 0] = preds[:, :, 0] * 8
            gt_poses[:, :, 1] = preds[:, :, 1] * 8
            gt_poses[:, :, 2] = maxval[:, :, 0]
        else:
        '''
        poses = targets[i]['cd']
        result_window = np.array([ori_size[0], ori_size[1], 1])
        gt_poses = np.copy(poses) * result_window

        img_h, img_w = img_size
        boxes = torch.mul(boxes, img_h)

        if vis and imgs is not None:
            img = imgs[i]
            gt_img = cv2.resize(img, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_AREA)
            pred_img = gt_img.copy()
            blank_img = np.zeros((ori_size[1], ori_size[0], 3), dtype = np.uint8)
            #print("blank_img ", blank_img.shape)
           
        gt_pose_to_draw = []
        for j in range(num_obj):
            label, conf = targets[i]['labels'][j].item(), 1
            x1, y1, x2, y2 = boxes[j]
            gt_pose_to_draw.append(gt_poses[j])
            #print(gt_poses[j])
            #x1, y1, x2, y2 = box_ops.sanitize_all_coordinates(targets[i]['boxes'][j][0], targets[i]['boxes'][j][1], targets[i]['boxes'][j][2], targets[i]['boxes'][j][3], img_size[0])
            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]
            pose_info = [img_id, gt_poses[j], mask_size[j], (x1.item(), y1.item(), x2.item(), y2.item())]

            groundtruths.append(pose_info)
            if label not in classes:
                classes.append(label)
            if False:
            #if vis:
                x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                rand_num = (img_id*x1)%19
                color= COLORS[rand_num]
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 1)
                label_name = COCO_CLASSES[label]
                text_str = '%s: %.2f' % (label_name, conf)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale, font_thickness = 0.6, 1
                
                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(gt_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)  # draw bbox
                cv2.putText(gt_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA) # append label and confidence score
                cv2.putText(gt_img, 'GT', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        if vis and i == 0:
            imshow_keypoints(gt_img, gt_pose_to_draw, skeleton, kpt_score_thr=0, \
                     pose_kpt_color=pose_kpt_color, pose_link_color=pose_link_color, radius=4, thickness=2)

        # Prediction Data
        pred = results[i]
        pred_score = pred['scores']
        pred_label = pred['labels']
        pred_boxes = pred['boxes']

        res_pose_ = res_pose[i]#.cpu().numpy()
        if pose_method != 'hm':
            result_window = np.array([ori_size[0]/dr_size, ori_size[1]/dr_size, 1])
            res_pose_ = np.copy(res_pose_) * result_window
        pose_to_draw = []
        num_obj = 0
        for q in range(pred_boxes.shape[0]):
            label, conf = pred_label[q].item(), pred_score[q].item()
            x1, y1, x2, y2 = pred_boxes[q]
            box_info = [img_id, label, conf, (x1.item(), y1.item(), x2.item(), y2.item())]

            pose_info = [img_id, res_pose_[q], conf]
            #print(res_pose_[q])
            if pred_score[q] > boxThrs:
                detections.append(pose_info)
            if label not in classes:
                classes.append(label)
            if vis:
                if pred_score[q] > boxThrs:
                    num_obj += 1
                    pose_to_draw.append(res_pose_[q])
                    
                    x1, y1, x2, y2 = x1.int().item(), y1.int().item(), x2.int().item(), y2.int().item()
                    rand_num = (img_id*x1)%19
                    color= COLORS[rand_num]
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 1)
                    label_name = COCO_CLASSES[label]
                    text_str = '%s %d: %.2f' % (label_name, q, conf)
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale, font_thickness = 0.6, 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(pred_img, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(pred_img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(pred_img, 'pred', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    
        if vis and i == 0:
            #imshow_keypoints(pred_img, pose_to_draw, skeleton, kpt_score_thr=0, \
            imshow_keypoints(blank_img, pose_to_draw, skeleton, kpt_score_thr=0, \
                     pose_kpt_color=pose_kpt_color, pose_link_color=pose_link_color, radius=4, thickness=2)
            res = np.concatenate((pred_img, blank_img, gt_img), axis=1)
            cv2.imwrite(os.path.join(img_dir, str()) +'_{}_pose_num{}.png'.format(img_id, num_obj), res)

        #if not vis and i >=batch_size/10:
        #    break 


    dects = detections
    gts = groundtruths
    
    npos = len(gts)

    dects = sorted(dects, key = lambda conf : conf[2], reverse=True)

    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))

    det = Counter(cc[0] for cc in gts)

    for key, val in det.items():
        det[key] = np.zeros(val)

    #print(len(dects))
    avg_iou = 0
    avg_kpiou = np.zeros((13))
    for d in range(len(dects)):
        gt = [gt for gt in gts if gt[0] == dects[d][0]]
        iouMax = 0
        kpiouMax = None

        for j in range(len(gt)):
            np_pred = dects[d][1]
            np_gt = gt[j][1]
            gt_mask_size = gt[j][2]
            gt_box = gt[j][3]
            
            iou1, kp_iou = oks_iou(np_gt, np_pred, gt_mask_size, gt_box)
            
            #iou1 = iou(dects[d][3], gt[j][3])
            if iou1 > iouMax:
                iouMax = iou1
                kpiouMax = kp_iou
                jmax = j

        avg_iou += iouMax
        avg_kpiou += kpiouMax

        if iouMax >= IOUThreshold:
            if det[dects[d][0]][jmax] == 0:
                TP[d] = 1
                det[dects[d][0]][jmax] = 1
            else:
                FP[d] = 1
        else:
            FP[d] = 1
    
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    #print(acc_TP, acc_FP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    if method == "AP":
        [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
    else:
        [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

    r = {
        'class' : 0,
        'precision' : prec,
        'recall' : rec,
        'AP' : ap,
        'iou' : avg_iou / len(dects),
        'kpiou' : avg_kpiou / len(dects),
        'interpolated precision' : mpre,
        'interpolated recall' : mrec,
        'total positives' : npos,
        'total TP' : np.sum(TP),
        'total FP' : np.sum(FP)
    }
    #print(r)
    result.append(r)

    return result


def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']

    #print("mAP len(result) = ", len(result))
    mAP = ap / len(result) if len(result) != 0 else 0.
    return mAP

def mIOU(result):
    iou = 0
    for r in result:
        iou += r['iou']

    mIOU = iou / len(result) if len(result) != 0 else 0.
    return mIOU

def mkpIOU(result):
    iou = np.zeros((13))
    for r in result:
        iou += r['kpiou']

    mIOU = iou / len(result) if len(result) != 0 else 0.
    return mIOU

def class_ap(result, c):
    ap = 0
    k = 0
    for r in result:
        if r['class'] == c:
            ap += r['AP']
            k +=1
    
    if k == 0:
        mAP = 0
    else:
        mAP = ap / k

    return mAP


