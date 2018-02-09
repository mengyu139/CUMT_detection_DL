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
from visdom import Visdom
import cv2
import pickle

from Net.ssd.utils.data_read import CustomDataset
from  Net.ssd.utils.visualize import get_img,show,nms,eval_img
from  Net.ssd.ssd_vgg.ssd_vgg_base import SSD_Net,make_vgg_layers,make_ectra_layers,make_loc_conf_layers,cfg,box_cfg,extras_cfg
from Net.ssd.loss import loss as Loss
from Net.ssd.utils.read_tmp import read_temperature


import matplotlib.pyplot as plt


if __name__ =="__main__":


    s_pickle=open('save.pickle','rb')
    Info=pickle.load(s_pickle)
    s_pickle.close()

    COST=0
    COST_CNT=0

    print(Info[0].__len__(),Info[1].__len__())

    X=[]
    Y=[]
    for t_i in range(40):
        threshould=(t_i+1)*0.025

        RECALL=0
        PERCISION=0
        for i in range(Info[0].__len__()):
            predict=Info[0][i]
            gt=np.array(Info[1][i])

            recall,percision=eval_img(result=predict,gt=gt,threshold=threshould)
            RECALL+=recall
            PERCISION+=percision

        X.append(RECALL/1./Info[0].__len__())
        Y.append(PERCISION/1./Info[0].__len__())



    plt.plot(X,Y,'*')
    plt.show()
    #
    # if Use_imshow:
    #     img = get_img(car_img.numpy()[0])
    #     if result is not None:
    #         for i in range(result.shape[0]):
    #             pt=result[i]*300
    #             cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,255,0),
    #                           thickness=2
    #                           )
    #
    #     gt_box=np.array(Ori_GTS[0])
    #     for i in range(gt_box.shape[0]):
    #             pt=gt_box[i]*300
    #             cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,0,255),
    #                           thickness=1
    #                           )
    #
    #
    #     cv2.imshow('img',img)
    #
    #     cv2.waitKey(0)
    #
    #
    # print("TP is %d ,FP is %d ,FN is %d  " %(TP,FP,FN))



