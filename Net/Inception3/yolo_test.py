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
import cv2
import sys


from Net import net as net
from Utils import data_read as data_read
from Utils import read_tmp as read_tmp
from Net import loss as Loss


import Net.Inception3.inception



def  IOU(Reframe,GTframe):
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




# img [3,h,w] ndarray
def un_nomalize_img(img):


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = np.array(mean,dtype=np.float32)
    mean = mean.reshape([3,1,1])

    std = np.array(std,dtype=np.float32)
    std = std.reshape([3,1,1])

    x = img*std
    x = x+mean

    # [c,h,w]->[h,w,c]
    x = x.transpose([1,2,0])

    x = cv2.cvtColor(x,code=cv2.COLOR_RGB2BGR)

    return x



# 把输出的 [x y w h] -> [x1 y1 x2 y2]的 box [8,7,7]
# box ndarray
def transform_box(box):
    # box = torch.autograd.Variable( torch.FloatTensor(8,7,7)).cuda()
    # box[...]=out_box[...]

    # [8,7,7] [x1 x1 y1 y1 x2 x2 y2 y2]
    result = np.zeros([8,7,7])

    result[0:2,...]=box[0:2,...]-0.5*box[4:6,...]# x1 x1
    result[2:4,...]=box[2:4,...]-0.5*box[6:8,...]# y1 y1

    result[4:6,...]=box[0:2,...]+0.5*box[4:6,...]# x2 x2
    result[6:8,...]=box[2:4,...]+0.5*box[6:8,...]# y2 y2

    return result


# output [14,7,7] ndarray
# img[3,448,448] ndarray
# gt_for_one_img
def test_one_img(output,img,gt_for_one_img, img_name,label=0):

    img = un_nomalize_img(img)

    # [8,7,7]
    box = transform_box(output[CLASS_NUM:CLASS_NUM+8,...])
    box = np.clip(box,0,1)

    # [4,7,7]
    cls=output[0:CLASS_NUM,...]
    # [1,7,7]
    cls = cls[label:label+1,...]

    # [7,7]
    # cls=np.argmax(cls,axis=0)

    # [2,7,7]
    conf_box=output[CLASS_NUM+8:CLASS_NUM+8+2,...]

    valid_conf_box=conf_box*np.tile(cls,[2,1,1])


    a=np.array([[0,0,0,0,0]])
    for i in range(7):#h
        for j in range(7):#w
            for b in range(2):
                x1=box[b,i,j]
                y1=box[2+b,i,j]
                x2=box[4+b,i,j]
                y2=box[6+b,i,j]
                conf=valid_conf_box[b,i,j]
                a=np.concatenate([a,np.array([[ x1,y1,x2,y2,conf ]])])

    a=a[1:,...]
    sort_arg = np.argsort(a[:,4]) #按第'4'列排序
    sort_a = a[sort_arg]
    sort_a = np.flip(sort_a,axis=0)

    print('img_name: ',img_name)

    for gt in gt_for_one_img:
        x1=int((gt[0]-0.5*gt[2])*img.shape[1])
        y1=int((gt[1]-0.5*gt[3])*img.shape[0])

        x2=int((gt[0]+0.5*gt[2])*img.shape[1])
        y2=int((gt[1]+0.5*gt[3])*img.shape[0])

        cv2.rectangle(img,pt1=(x1,y1),
                      pt2=(x2,y2),thickness=3,
                      color=(255,0,0),)

    # 初步筛选
    k1=0.1
    sort_a = sort_a[ sort_a[:,4]>k1,... ]

    if sort_a.shape[0]> 0:
        #NMS
        k2=0.5
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

        print('filtered_box  size is: ',filtered_box.__len__())


        for c in range(filtered_box.__len__()):
            choose_box=filtered_box[c]
            choose_box *= img.shape[0]

            cv2.rectangle(img,pt1=(int(choose_box[0]),int(choose_box[1])),
                          pt2=(int(choose_box[2]),int(choose_box[3])),
                          color=(0,255,0),
                          thickness=3
                  )


    cv2.imshow('img',img)
    flag=True
    while flag:

        key=cv2.waitKey(30)&0xff

        if key ==110:
            flag=False


    return a

    #find predict box corresponding to label





CLASS_NUM=1
train_btach_size=1
save_mode_name = 'yolo_inception.pth'
img_root = '/home/jason/Dataset/VOCdevkit_2007_test/VOC2007/JPEGImages/'
Use_cuda = True

# =======Setup net==================================
Inception3_model = Net.Inception3.inception.Inception3()
yolo_net = Net.Inception3.inception.Inception_yolo(Inception3_model,is_cuda=Use_cuda,class_num=CLASS_NUM)


# =======Load parameters==================================
save_dict = torch.load( save_mode_name )
yolo_net.load_state_dict(save_dict)


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

optimizer = torch.optim.SGD([
    # {'params':train_para ,'lr':0.001},
    {'params':yolo_net.features.parameters(),'lr':0.0001},
    {'params':yolo_net.extra_conv.parameters(),'lr':0.0001},
    {'params':yolo_net.fc_layer.parameters(),'lr':0.0001,'weight_decay':0.0001},

],momentum=0.9,weight_decay=0.0)
# =========================================


#================set up data reader===============================================================
train_dataset = data_read.CustomDataset(img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/testV3.txt',\
                                        is_train=False,label_map={'car':0},class_num=1,
                                        img_size=585)
# test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
data_loader={}
data_loader["train"]=torch.utils.data.DataLoader(train_dataset, batch_size=train_btach_size,shuffle=True, num_workers=4)
# data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)
# =========================================

# viz = Visdom()
# # line updates
# Vis_loss = viz.line(
# X=np.array([0]),
# Y=np.array([0]),
# )
dis_cnt=0


yolo_net.eval()
for epoch in range(1000):
    GPU_TEM = read_tmp.read_temperature()
    print ('+++++++++++++++++++++++++++++++++++++gpu tem :',GPU_TEM)
    if GPU_TEM > 80:
        print ('GPU OVER HEATED,QUIT!!!')
        break

    COST=0
    CNT=0
    for item in data_loader["train"]:
        dis_cnt+=1
        CNT += 1
        imgs,indexs,img_names,flips = item
        GT=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
        YOLO_GT = train_dataset.get_YOLO_GTS(GT)

        optimizer.zero_grad()

        if Use_cuda:
            train_x = torch.autograd.Variable(imgs,volatile=True).cuda()
        else:
            train_x = torch.autograd.Variable(imgs,volatile=True)


        result = yolo_net(train_x)

        # print(imgs.size() , result.size())

        # loss = Loss.loss_for_batch(result,YOLO_GT)
        #
        # COST+=loss.cpu().data.numpy()[0]
        #
        # if CNT % 5==0:
        #     sys.stdout.write('.')
        #     sys.stdout.flush()
        #
        if Use_cuda:
            outputs=result.cpu().data.numpy()
        else:
            outputs=result.data.numpy()
        #
        #
        a=test_one_img(outputs[0],imgs.numpy()[0],YOLO_GT[0],img_names[0],label=0)

        loss = Loss.loss_for_batch(result,YOLO_GT)
        # a=test_one_img(outputs[0],imgs.numpy()[0],YOLO_GT[0],label=1)
        # a=test_one_img(outputs[0],imgs.numpy()[0],YOLO_GT[0],label=2)
        # a=test_one_img(outputs[0],imgs.numpy()[0],YOLO_GT[0],label=3)



    # print(' ')
    # print('---------------------cost is: ',COST/1./CNT)
    #
    # torch.save(yolo_net.state_dict(),save_mode_name)
    # print('save modle in ',save_mode_name )