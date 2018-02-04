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
        #