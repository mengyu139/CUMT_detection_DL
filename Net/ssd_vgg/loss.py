# coding=utf-8

import torch.nn as nn
import torch.nn.functional
import torch.autograd
import torch.tensor
import torch.optim
import torchvision
import torch.utils.data
import torchvision.transforms

import math
import numpy as np
import sys

from Net.ssd_vgg.utils.data_read import CustomDataset
from  Net.ssd_vgg.utils.visualize import get_img,show

# box_a box_b ndarray [num,4]  [x1 y1 x2 y2]
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

# matched priors [x y w h]
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = matched[:,0:2]- priors[:, 0:2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = matched[:,2:4] / priors[:, 2:4]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes#[x1 y1 x2 y2]



# priors    Variable (CPU requires_grad False)      [8732,4]   [x y w h]
# GT        Variable (CPU requires_grad False)    [gt_num,5]      gt0:[x y w h,cls] cls[ 0:background 1~n : frontground  ]
# return loc_t,conf Variable (requires_grad False)
# [8732,4]  [8732,]
def clac_target_for_one_img(priors,GT,threshold=0.5):

    # print( 'GT size: ',GT.size() )
    GT_loc = GT[:,0:4]
    GT_cls = GT[:,4]

    # [gt_num, 8732] named[A ,B]
    overlaps = jaccard(
        point_form(GT_loc),#[gt_num,4] xywh -> x1 y1 x2 y2
        point_form(priors)#[8732,4]     xywh -> x1 y1 x2 y2
    )

     # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)


    # # 保证 每个GT 都有一个对应的 prior ，对于剩下的prior，找出最大的作为 positive，其余的都作为negtive
    # # 这样，为8732个box 每个都找到 其match的 gt（如果可以match）
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    #
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    #
    matches = GT_loc[best_truth_idx]    #[8732,4]
    conf = GT_cls[best_truth_idx] + 1   #label start from 1

    conf[best_truth_overlap < threshold] = 0 #  0 means background

    loc_t = encode(matches, priors, variances=[0.1, 0.2])

    # print('loc_t size: ',loc_t.size(),'conf size',conf.size())
    return loc_t,conf


# priors    Variable (CPU requires_grad False)      [8732,4]       [x y w h]
# GT        Variable (CPU requires_grad False)    [gt_num,5]      gt0:[x y w h,cls] cls[ 0:background 1~n : frontground  ]
# out
# out[0]    Variable  [8732,4]
# out[1]    Variable  [8732,num_classes]

# return loc_loss,conf_loss
def loss_for_one_img(priors,GT,out,threshold=0.5,k=3,is_cuda=False):
    # [8732,4]  [8732]
    loc_t,conf_t=clac_target_for_one_img(priors,GT,threshold)

    if is_cuda:
        loc_t=loc_t.cuda()
        conf_t=conf_t.cuda().type(torch.cuda.LongTensor)
    else:
        conf_t=conf_t.type(torch.LongTensor)

    # [8732,4]  [8732,classes=2]
    loc_out = out[0]
    conf_out = out[1]

    pos_mask = conf_t>0
    neg_mask = conf_t==0

    pos_num = conf_t.data.sum()

    # loc loss
    loc_loss = torch.nn.functional.smooth_l1_loss(input=loc_out,target=loc_t,
                                                             size_average=False,reduce=False)

    loc_loss = loc_loss * pos_mask.unsqueeze(1).expand_as(loc_out).float()
    loc_loss = loc_loss.sum()

    # conf loss = conf_postive  + conf_negtive
    # [8732]
    conf_cost=torch.nn.functional.cross_entropy(input=conf_out,target=conf_t,size_average=False,reduce=False)

    conf_postive = conf_cost * pos_mask.float()
    conf_postive = conf_postive.sum()

    conf_cost_neg = conf_cost[neg_mask]
    conf_cost_neg,_ = conf_cost_neg.sort(descending=True)

    select_num = min( loc_t.size(0)-pos_num,k*pos_num )

    if select_num != 0:
        conf_cost_neg=conf_cost_neg[0:select_num]
        conf_negtive=conf_cost_neg.sum()

        conf_loss = conf_postive + conf_negtive
        conf_loss *= 1.
    else:
        conf_loss=0

    return loc_loss/pos_num,conf_loss/pos_num


# priors    Variable (volatile)    [8732,4]       [x y w h]
# GT        Variable (volatile)    [gt_num,5]      gt0:[x y w h,cls] cls[ 0:background 1~n : frontground  ]
# outputs   List [output1 output2]   [b,8732,4]  [b,8732,cls=2]
# return Loss
def loss_for_batch(priors,GTS,outputs,threshold=0.5,k=3,is_cuda=False):

    if is_cuda:
        Loss = torch.autograd.Variable(torch.zeros(outputs[0].size(0))).cuda()
    else:
        Loss = torch.autograd.Variable(torch.zeros(outputs[0].size(0)))

    for i in range( outputs[0].size(0) ):

        GT = torch.autograd.Variable( torch.FloatTensor( GTS[i] ) )

        loss1,loss2=loss_for_one_img(priors,GT,[outputs[0][i],outputs[1][i]],threshold,k,is_cuda)
        Loss[i]=loss1+loss2

    return Loss.mean()



if __name__ =="__main__":

    priors=torch.autograd.Variable(torch.FloatTensor(np.zeros([8732,4])))
    GT=torch.autograd.Variable(torch.FloatTensor(np.zeros([10,5])))

    clac_target_for_one_img(priors,GT,threshold=0.5)


    # img_root = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'
    # train_btach_size=1
    #
    # train_dataset = CustomDataset(img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/trainV3.txt',\
    #                                     is_train=True,label_map={'car':0},class_num=2,
    #                                     img_size=300)
    # # test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
    # data_loader={}
    # data_loader["train"]=torch.utils.data.DataLoader(train_dataset, batch_size=train_btach_size,shuffle=False, num_workers=1)
    # # data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)
    # # =========================================
    #
    # for item in data_loader['train']:
    #
    #     imgs,indexs,img_name,flips = item
    #
    #     GTS=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
    #     GTS=train_dataset.get_SSD_GTS(GTS)
    #     print(imgs.size())
    #
    #     img = get_img(imgx=imgs.numpy()[0])
    #
    #     show(img,GTS[0])