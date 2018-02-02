# coding=utf-8

import torch.nn as nn
import torch.nn.functional
import torch.autograd
import torch.tensor
import torch.optim
import torchvision
import torch.utils.data
import torchvision.transforms

from visdom import Visdom
import math
import numpy as np
import PIL.Image
import os
import time
import json
import pickle

from Net import net as net


# pre_boxs:[8,7,7] Variable
# gt_boxs:[8,7,7]  Variable

# 把 GT的 box 从 xywh 形式 转换成x1 y1 x2 y2 形式
# box:Variable
def transform_gt_box(box):

    result = torch.autograd.Variable( torch.FloatTensor(np.zeros([8,7,7])) ).cuda()
    result[0:2,...]=box[0:2,...]-0.5*box[4:6,...]# x1 x1
    result[2:4,...]=box[2:4,...]-0.5*box[6:8,...]# y1 y1

    result[4:6,...]=box[0:2,...]+0.5*box[4:6,...]# x2 x2
    result[6:8,...]=box[2:4,...]+0.5*box[6:8,...]# y2 y2

    return result

# 把输出的 [x y w h] -> [x1 y1 x2 y2]的 box [8,7,7]
# box Variable
def transform_output_box(box):

    # box = torch.autograd.Variable( torch.FloatTensor(8,7,7)).cuda()
    # box[...]=out_box[...]

    # [8,7,7] [x1 x1 y1 y1 x2 x2 y2 y2]
    result = torch.autograd.Variable( torch.FloatTensor(8,7,7)).cuda()

    result[0:2,...]=box[0:2,...]-0.5*box[4:6,...]# x1 x1
    result[2:4,...]=box[2:4,...]-0.5*box[6:8,...]# y1 y1

    result[4:6,...]=box[0:2,...]+0.5*box[4:6,...]# x2 x2
    result[6:8,...]=box[2:4,...]+0.5*box[6:8,...]# y2 y2

    return result




def iou(pre_boxs,gt_boxs):
    box_for_every_ceil= int(pre_boxs.size(0)/4)

    iou_truth=torch.autograd.Variable( torch.FloatTensor(np.zeros([2,7,7])) ).cuda()

    PRE_bounding= transform_output_box( pre_boxs )
    GT_bounding= transform_gt_box( gt_boxs )

    for i in range( box_for_every_ceil ):
        x1_max = torch.cat( [PRE_bounding[i+0*2,...].unsqueeze(0),GT_bounding[i+0*2,...].unsqueeze(0)],dim=0)
        x1_max,_  = torch.max(x1_max,dim=0)

        y1_max = torch.cat( [PRE_bounding[i+1*2,...].unsqueeze(0),GT_bounding[i+1*2,...].unsqueeze(0)],dim=0)
        y1_max,_  = torch.max(y1_max,dim=0)

        x2_min  = torch.cat( [PRE_bounding[i+2*2,...].unsqueeze(0),GT_bounding[i+2*2,...].unsqueeze(0)],dim=0)
        x2_min,_  = torch.min(x2_min,dim=0)

        y2_min  = torch.cat( [PRE_bounding[i+3*2,...].unsqueeze(0),GT_bounding[i+3*2,...].unsqueeze(0)],dim=0)
        y2_min,_  = torch.min(y2_min,dim=0)

        delta_x = x2_min - x1_max
        delta_y = y2_min - y1_max

        mask1=delta_x>0
        mask2=delta_y>0

        intersetion = (x2_min - x1_max)*(y2_min - y1_max)

        intersetion = intersetion * mask1.float() * mask2.float()

        iou_truth[i,...]= intersetion/( pre_boxs[i+2*2,...]*pre_boxs[i+2*3,...]+gt_boxs[i+2*2,...]*gt_boxs[i+2*3,...]-intersetion )

    return iou_truth




def l2_loss(input,target,reduce=True):

    result = torch.pow(input-target,2)

    if reduce:
        return result.mean()
    else:
        return result



# net_out:[14,7,7]  14: [cls0 cls1 cls2 cls3] 2box[x x y y w w h h c c]
# yolo_gt: list [11] 11:x y w h class gx gy gx1 gy1 gx2 gy2
# return: loss[4]   coord_loss object_loss noobject_loss class_loss

def loss_for_one_GT(net_out,yolo_gt,class_num=1):
    box_num = 2

    x, y, w, h, cls, gx, gy, gx1, gy1, gx2, gy2=yolo_gt
    # ---------------------------------------------------------------
    # [2,7,7] Variable 表示 目标的中心出现的格子
    response = np.zeros([2,7,7])
    response[0:2,gy,gx]=1#注意， tensor in pytorch is NCHW
    response = torch.autograd.Variable(torch.FloatTensor(response)).cuda()

    # [1,7,7] Variable 表示目标出现的格子
    obj_mask = np.zeros([1,7,7])
    obj_mask[0,gy1:gy2+1,gx1:gx2+1]=1#注意， tensor in pytorch is NCHW
    obj_mask = torch.autograd.Variable(torch.FloatTensor(obj_mask)).cuda()

    # [8,7,7] Variable box [x x y y w w h h]
    gt_box = np.array([x,x,y,y, w,w, h,h])
    gt_box = gt_box.reshape([8,1,1])
    gt_box = np.tile(gt_box,[1,7,7])# to [8,7,7]
    gt_box = torch.autograd.Variable(torch.FloatTensor(gt_box)).cuda()

    pre_box=net_out[class_num:class_num+8,...]
    # ================
    # [2,7,7] Variable 预测框和 GT的 IOU
    iou_truth = iou(pre_box,gt_box)

    # [2,7,7] Variable  论文中的 1(obj i,j) 表示目标的中心出现的格子,并且iou值最大的那个box
    response_iou=iou_truth*response
    I_maxium,I_index=response_iou.max(dim=0,keepdim=True)
    mask = torch.ge(response_iou,I_maxium).float()
    I = mask * response

    no_I = torch.autograd.Variable(torch.ones([2,7,7])).cuda() - I

    # [2,7,7] Variable 预测 和 真实 的 confidence
    pre_confidence=net_out[class_num+8:class_num+8+2,...]
    gt_confidence = iou_truth*response

    # [class,7,7] Variable 预测 和 真实 的 classes
    pre_classes=net_out[0:class_num,...]
    cls_vec = [0]*class_num
    cls_vec[cls]=1
    gt_classes = np.array(cls_vec)
    gt_classes = gt_classes.reshape([class_num,1,1])
    gt_classes = np.tile(gt_classes,[1,7,7])
    gt_classes = torch.autograd.Variable(torch.FloatTensor(gt_classes)).cuda()

    # -----------------------------------------------------------------
    # (1)coord loss [2,7,7]
    # x , y
    coord_loss_1=l2_loss(pre_box[0:2],gt_box[0:2],reduce=False)+l2_loss(pre_box[2:4],gt_box[2:4],reduce=False)
    # w,h

    coord_loss_2=l2_loss(torch.sqrt(pre_box[4:6]),torch.sqrt(gt_box[4:6]),reduce=False)+\
                 l2_loss(torch.sqrt(pre_box[6:8]),torch.sqrt(gt_box[6:8]),reduce=False)

    coord_loss = (coord_loss_1 + coord_loss_2)*I

    # (2) confidence object_loss [2,7,7]
    object_loss = l2_loss(pre_confidence,gt_confidence,reduce=False)*I

    # (3)confidence noobject_loss [2,7,7]
    noobject_loss=l2_loss( pre_confidence,gt_confidence*I,reduce=False)*no_I

    # (4)class_loss
    #[4,7,7] -> [1,7,7]
    class_loss = l2_loss(pre_classes,gt_classes,reduce=False).sum(dim=0,keepdim=True)*obj_mask


    return [coord_loss.sum() ,object_loss.sum() ,noobject_loss.sum() ,class_loss.sum()]



def loss_for_one_img(net_out,gt_for_one_img,class_num=1):

    coord_loss = torch.autograd.Variable( torch.FloatTensor([0]) ).cuda()
    object_loss= torch.autograd.Variable( torch.FloatTensor([0]) ).cuda()
    noobject_loss= torch.autograd.Variable( torch.FloatTensor([0]) ).cuda()
    class_loss= torch.autograd.Variable( torch.FloatTensor([0]) ).cuda()

    for index,item in enumerate(gt_for_one_img):
        loss=loss_for_one_GT(net_out,yolo_gt=item,class_num=class_num)

        # print('loss_for_one_GT: ',loss.__len__(),loss[0].size(),coord_loss.size())
        coord_loss+=loss[0]
        object_loss+=loss[1]
        noobject_loss+=loss[2]
        class_loss+=loss[3]

    obj_num = gt_for_one_img.__len__()

    coord_loss =coord_loss * 5. / obj_num
    object_loss =object_loss * 1./ obj_num
    noobject_loss =noobject_loss * 0.5/ obj_num
    class_loss = class_loss * 1./ obj_num
    return [coord_loss,object_loss,noobject_loss,class_loss]


def loss_for_batch(net_batch_out,gt_batch,class_num=1):
    loss = torch.autograd.Variable( torch.zeros([net_batch_out.size(0),1]) ).cuda()
    for i in range( net_batch_out.size(0) ):
        loss_list = loss_for_one_img( net_batch_out[i],gt_batch[i],class_num )
        loss[i,0] = loss_list[0]+loss_list[1]+loss_list[2]+loss_list[3]

        # print( 'batch img: ',i,\
        #        ' coord_loss:',loss_list[0].cpu().data.numpy()[0],
        #        ' object_loss: ',loss_list[1].cpu().data.numpy()[0],
        #        ' noobject_loss: ',loss_list[2].cpu().data.numpy()[0],
        #        ' class_loss: ',loss_list[3].cpu().data.numpy()[0],
        #        )

    return loss.mean()




if __name__=="__main__":

    a=np.array([0.5,0.5,0.5,0.5,1,1,1,1])
    a = a.reshape([8,1,1])
    a = np.tile(a,[1,7,7])

    a = torch.autograd.Variable(torch.FloatTensor(a)).cuda()

    b=np.array([1,1.5,1,1.5,1,1,1,1])
    b = b.reshape([8,1,1])
    b = np.tile(b,[1,7,7])
    b = torch.autograd.Variable(torch.FloatTensor(b)).cuda()

    r = iou(a,b)


    print(r.size())

    print(r[0])
    print(r[1])


    # b=np.array([0,1,2,3,4,5,6])
    # b = np.reshape(b,[7,1])
    # b= np.tile(b,[1,7])
    # b = b[np.newaxis,...]
    # b=np.tile(b,[2,1,1])
    #
    #
    # print(b.shape)
    #
    # print(b)
    # a = torch.autograd.Variable( torch.FloatTensor(8,7,7)).cuda()
