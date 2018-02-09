# coding=utf-8

import torch.nn
import torch.nn.functional
import torch.autograd
import torch.tensor
import torch.optim
import torchvision
import torch.utils.data
import torchvision.transforms

from visdom import Visdom
# from graphviz import Digraph

import math
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image,PIL.ImageEnhance
import os
import time
import json
import collections
import random
import cv2



def get_img(imgx):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = np.array(mean,dtype=np.float32)
    mean = mean.reshape([3,1,1])

    std = np.array(std,dtype=np.float32)
    std = std.reshape([3,1,1])

    img = imgx.copy()
    img = img*std
    img = img+mean

    img=np.transpose(img,[1,2,0])
    # print(img.shape,type(img))

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # print(img.shape)
    return img



def show(img,gts):
    W=img.shape[1]
    H=img.shape[0]

    for i,gt in enumerate(gts):

        pt1_x = int(W*(gt[0]-0.5*gt[2]))
        pt1_y = int(H*(gt[1]-0.5*gt[3]))
        pt2_x = int(W*(gt[0]+0.5*gt[2]))
        pt2_y = int(H*(gt[1]+0.5*gt[3]))


    #     print(pt1_x,pt1_y,pt2_x,pt2_y)
    #     img = np.array(img)
    #
    #     print('img type: ',type(img),img.shape)
    #     cv2.imshow('img',img)
    #     cv2.waitKey(0)
        cv2.rectangle(img,pt1=(pt1_x,pt1_y),pt2=(pt2_x,pt2_y),color=(0,255,0),thickness=2)


    cv2.imshow('img',img)
    cv2.waitKey(0)


def IOU(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。·
    """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0
    else:
        Area = width*height # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio

# box  ndarray [n,5] [x1 y1 x2 y2 conf]
def nms(box,threshold=0.5):
    if box.shape[0] == 1:
        return box
    else:
        sort_arg = np.argsort(box[:,4]) #按第'4'列排序
        box = box[sort_arg]
        box = np.flip(box,axis=0)

        # 初步筛选
        k1=0.4
        sort_a = box[ box[:,4]>k1,... ]

        if sort_a.shape[0]> 0:
            #NMS
            k2=threshold
            pro_filter=[1]*sort_a.shape[0]

            for i in range(sort_a.shape[0]):
                if pro_filter[i] == 0:
                    continue
                else:
                    for j in range(i+1,sort_a.shape[0]):
                        intersect = IOU(sort_a[i,0:4],sort_a[j,0:4])

                        if intersect > k2:
                            pro_filter[j]=0

            filtered_box = []
            for i in range( sort_a.shape[0] ):
                if pro_filter[i] is not 0:
                    filtered_box.append(sort_a[i])

            return np.array(filtered_box)

        else:
            return None

# result ndarray [n,4]
# gt ndarray [g,4]
def eval_img(result,gt,threshold):
    tp=0
    fp=0
    fn=0

    if result is None:
        tp=0
        fp=0
        fn=gt.shape[0]
        recall=0
        percision=0
        # return tp,fp,fn
    else:
        result=result[0:4]
        gt=gt[0:4]
        for i in range(gt.shape[0]):
            for j in range(result.shape[0]):
                if IOU(gt[i],result[j])>=threshold:
                    tp+=1
                    break

        percision=tp/1./result.shape[0]
        recall=tp/1./gt.shape[0]

    return recall,percision


if __name__=="__main__":
    # a=[0.075,0.345,0.915,0.784]
    # b=[0.0847,0.335,0.922,0.776]
    #
    # c=IOU(b,a)
    #
    # print(c)
    a=np.array([[1,2,3,4,0],[0,1,2,3,2],[3,4,5,6,1],[4,5,6,7,4],[5,5,5,5,7]])
    x=nms(a)

    print(x)