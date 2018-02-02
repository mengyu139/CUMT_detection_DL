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
import sys
import cv2

import sys
sys.path.append('/home/jason/PycharmProjects/CUMT_YOLO')

import Utils.data_read as data_read
import Net.loss as Loss
import Utils.read_tmp as read_tmp
# from Utils import data_read as data_read
# from Utils import read_tmp as read_tmp
# from Net import loss as Loss


import Net.Inception3.inception



train_btach_size=10
save_mode_name = 'yolo_inception.pth'
img_root = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'
Use_cuda = True

# =======Setup net==================================
Inception3_model = Net.Inception3.inception.Inception3()
yolo_net = Net.Inception3.inception.Inception_yolo(Inception3_model,is_cuda=Use_cuda,class_num=1)


# =======Load parameters==================================
save_dict = torch.load( save_mode_name )
yolo_net.load_state_dict(save_dict)
#
# save_dict = torch.load( '/home/jason/PycharmProjects/CUMT_YOLO/Model/inception_v3_google-1a9a5a14.pth' )
# state_dict = yolo_net.state_dict()
# key_list=[]
# for key in save_dict:
#     key_list.append('features.'+key)
#
# for key in state_dict:
#     if key in key_list:
#         state_dict[key]=save_dict[key.lstrip('features.')]
# yolo_net.load_state_dict(state_dict)
# =========================================


#================set up optimiser===============================================================
yolo_net.train()
if Use_cuda:
    yolo_net.cuda()
else:
    yolo_net.cpu()

k=0.0005
optimizer = torch.optim.SGD ([
    # {'params':train_para ,'lr':0.001},
    {'params':yolo_net.features.parameters(),'lr':k},
    {'params':yolo_net.extra_conv.parameters(),'lr':k},
    {'params':yolo_net.fc_layer.parameters(),'lr':k},

],weight_decay=0.0,momentum=0.9)
# =========================================


#================set up data reader===============================================================

train_dataset = data_read.CustomDataset(img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/trainV3.txt',\
                                        is_train=True,label_map={'car':0},class_num=1,
                                        img_size=585)
# test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
data_loader={}
data_loader["train"]=torch.utils.data.DataLoader(train_dataset, batch_size=train_btach_size,shuffle=True, num_workers=8)
# data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)
# =========================================

viz = Visdom()
# line updates
Vis_loss = viz.line(
X=np.array([0]),
Y=np.array([0]),
)
dis_cnt=0



for epoch in range(500):

    GPU_TEM = read_tmp.read_temperature()
    print ('+++++++++++++++++++++++++++++++++++++gpu tem :',GPU_TEM)
    if GPU_TEM > 80:
        print ('GPU OVER HEATED,QUIT!!!')
        break

    COST=0
    CNT=0
    yolo_net.train()

    for item in data_loader["train"]:
        dis_cnt+=1
        CNT += 1
        imgs,indexs,img_names,flips = item
        GT=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
        YOLO_GT = train_dataset.get_YOLO_GTS(GT)

        optimizer.zero_grad()

        if Use_cuda:
            train_x = torch.autograd.Variable(imgs).cuda()
        else:
            train_x = torch.autograd.Variable(imgs)

        result = yolo_net(train_x)

        # print(imgs.size() , result.size())

        loss = Loss.loss_for_batch(result,YOLO_GT)

        viz.line(
            X=np.array([dis_cnt]),
            Y=np.array( [loss.cpu().data.numpy()[0]] ), win=Vis_loss, update='append')

        COST+=loss.cpu().data.numpy()[0]

        if CNT % 2==0:
            sys.stdout.write('.')
            sys.stdout.flush()

        loss.backward()
        optimizer.step()

    print(' ')
    print('---------------------cost is: ',COST/1./CNT)

    if epoch % 5 == 0:
        torch.save(yolo_net.state_dict(),save_mode_name)
        print('save modle in ',save_mode_name )