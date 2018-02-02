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
import sys

from Net import net as net
from Utils import data_read as data_read
from Utils import read_tmp as read_tmp
from Net import loss as Loss

if __name__=="__main__":

    save_mode_name = 'yolo_vgg_car.pth'
    #     ================set up net===============================================================
    yolo_net = net.VGG(net.make_layers(net.cfg['D'], batch_norm=True), num_classes=4)

    save_dict = torch.load( save_mode_name )
    yolo_net.load_state_dict(save_dict)

    # save_dict = torch.load( '/home/jason/PycharmProjects/CUMT_YOLO/Model/vgg16_bn-6c64b313.pth' )
    # state_dict = yolo_net.state_dict()
    # # #
    # for key in save_dict:
    #     print(key)
    #     if  'features' in key:
    #         state_dict[key] = save_dict[key]
    #
    # # #!!!!!!!!!!!!!!!!!!!!!load save paraments
    # yolo_net.load_state_dict(state_dict)
    # ================================================

    # ================set para state in net=========================
    modulde_cnt=0
    train_para = []
    freeze_para=[]

    for child in yolo_net.features.children():
        print ('module is: ',modulde_cnt,' : ',child," para num : ",list(child.parameters()).__len__(),)
        if modulde_cnt < 0:
            if isinstance(child,torch.nn.BatchNorm2d):
                train_para.extend(child.parameters() )
                print (' train')
            else:
                freeze_para.extend(child.parameters() )
                for param in child.parameters():
                    param.requires_grad = False

                print (' freeze')
        else:
            train_para.extend(child.parameters() )
            print (' train')

        # add hook
        if modulde_cnt == 43:
            pass
            # print ('add hook to ',child)
            # child.register_forward_hook(hook=hook)

        modulde_cnt+=1
        # print train_para.__len__() , freeze_para.__len__()

      #================set up optimiser===============================================================
        yolo_net.train()
        yolo_net.cuda()

        optimizer = torch.optim.SGD([
            # {'params':train_para ,'lr':0.001},
            {'params':yolo_net.features.parameters(),'lr':0.0005},
            {'params':yolo_net.extra_conv.parameters(),'lr':0.0005},
            {'params':yolo_net.fc_layer.parameters(),'lr':0.0005,'weight_decay':0.0005},

        ],momentum=0.9,weight_decay=0)

    train_btach_size=8
    img_root = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'

    train_dataset = data_read.CustomDataset(img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/trainV2.txt',\
                                            is_train=True,label_map={'person':0,'car':1,'bus':2,'bicycle':3},img_size=448)
    # test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
    data_loader={}
    data_loader["train"]=torch.utils.data.DataLoader(train_dataset, batch_size=train_btach_size,shuffle=True, num_workers=2)
    # data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)

    viz = Visdom()
    # line updates
    Vis_loss = viz.line(
    X=np.array([0]),
    Y=np.array([0]),
    )
    dis_cnt=0

    for epoch in range(1000):
        GPU_TEM = read_tmp.read_temperature()
        print ('+++++++++++++++++++++++++++++++++++++gpu tem :',GPU_TEM)
        if GPU_TEM > 80:
            print ('GPU OVER HEATED,QUIT!!!')
            break

        # img,index,img_name,flip
        yolo_net.train()

        COST=0
        CNT=0
        for item in data_loader["train"]:
            dis_cnt+=1
            CNT += 1
            imgs,indexs,img_names,flips = item
            GT=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
            YOLO_GT = train_dataset.get_YOLO_GTS(GT)

            optimizer.zero_grad()

            train_x = torch.autograd.Variable(imgs).cuda()
            _,_ ,out3= yolo_net(train_x)

            # print('img_names : --------------',img_names)
            sys.stdout.write('.')
            sys.stdout.flush()
            loss = Loss.loss_for_batch(out3,YOLO_GT)

            COST+=loss.cpu().data.numpy()[0]

            viz.line(
            X=np.array([dis_cnt]),
            Y=np.array( [loss.cpu().data.numpy()[0]] ), win=Vis_loss, update='append')


            loss.backward()
            optimizer.step()
        print(' ')
        print('---------------------cost is: ',COST/1./CNT)

        torch.save(yolo_net.state_dict(),save_mode_name)
        print('save modle in ',save_mode_name )

            # print(loss.cpu().data.numpy()[0])