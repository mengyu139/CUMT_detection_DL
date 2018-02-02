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
import numpy as np
import PIL.Image

import cv2

from Utils import data_read as data_read
from Net import loss as loss


def plot_grid(img,grid_list):

    for item in grid_list:

        x=int((item[0]+0.5)/7.*448)
        y=int((item[1]+0.5)/7.*448)

        cv2.circle(img,center=(x,y),radius=10,color=(0,10,255),thickness=-1)


def plot_grid_mask(img,grid_list):

    x1= int(grid_list[0][0]/7.*448)
    y1= int(grid_list[0][1]/7.*448)
    x2= int((grid_list[1][0]+1)/7.*448)
    y2= int((grid_list[1][1]+1)/7.*448)

    cv2.rectangle(img,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=2)


    # mask = np.ones_like(img[y1:y2,x1:x2,...])*100
    #
    # s=cv2.addWeighted(src1=mask,alpha=0.5,src2=img[y1:y2,x1:x2,...],beta=0.5,gamma=1)
    #
    # s = s*255
    #
    # s = s.astype(np.uint8)
    #
    # img[y1:y2,x1:x2,...] = s[...]
    # print(1)



def show_image_and_GT(imgs,gts=None,flips=None):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean = np.array(mean,dtype=np.float32)
    mean = mean.reshape([1,3,1,1])

    std = np.array(std,dtype=np.float32)
    std = std.reshape([1,3,1,1])

    imgs = imgs*std
    imgs = imgs+mean

    for i in range( imgs.shape[0] ):

        vis_x= imgs[i]
        vis_x = vis_x.transpose([1,2,0])
        vis_x = cv2.cvtColor(vis_x,code=cv2.COLOR_RGB2BGR)

        W=vis_x.shape[1]
        H=vis_x.shape[0]

        color_list=[(255,0,0),(0,255,0),(0,0,255),(0,0,0)]

        for j in range( gts[i].__len__() ):

            pt1_x=0
            pt1_y=0
            pt2_x=0
            pt2_y=0

            x=gts[i][j][0]
            y=gts[i][j][1]
            w=gts[i][j][2]
            h=gts[i][j][3]

            pt1_x = int((x-0.5*w)*W)
            pt1_y = int((y-0.5*h)*H)
            pt2_x = int((x+0.5*w)*W)
            pt2_y = int((y+0.5*h)*H)


            cls =  gts[i][j][4]

            #
            # cv2.rectangle(vis_x,pt1=(pt1_x,pt1_y)\
            #               ,pt2=(pt2_x,pt2_y),\
            #               color=color_list[cls],thickness=2)

            plot_grid(vis_x,[ [gts[i][j][5],gts[i][j][6]]  ]  )#,[gts[i][j][7],gts[i][j][8]],[gts[i][j][9],gts[i][j][10]]
            # cv2.circle(vis_x,center=( int(W*gts[i][j][4]),int(H*gts[i][j][5]) ),\
            #            radius=3,color=(0,255,0),thickness=-1)

            plot_grid_mask( vis_x,[ [gts[i][j][7],gts[i][j][8]],[gts[i][j][9],gts[i][j][10]] ] )



        cv2.imshow('img',vis_x)
        print(flips[i])

        cv2.waitKey(3000)





if __name__=="__main__":

    f = open('/home/jason/PycharmProjects/CUMT_YOLO/Dataset/train.txt','r')
    lines = f.readlines()
    f.close()

    print(lines.__len__())


    train_btach_size=1
    img_root = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'


    train_dataset = data_read.CustomDataset(img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/train.txt',\
                                            is_train=True,label_map={'person':0,'car':1,'bus':2,'bicycle':3})
    # test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
    data_loader={}


    data_loader["train"]=torch.utils.data.DataLoader(train_dataset, batch_size=train_btach_size,shuffle=True, num_workers=2)
    # data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)


    for item in data_loader["train"]:

        img,indexs,img_names,flips = item

        # print(img.size())
        if img.size(1) is not 3:
            print('error')

        GT=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
        YOLO_GT = train_dataset.get_YOLO_GTS(GT)

        show_image_and_GT(img.numpy(),gts=YOLO_GT,flips=flips)

        # sim_out = torch.rand([train_btach_size,14,7,7])
        # sim_out = torch.autograd.Variable(sim_out,requires_grad=True).cuda()
        #
        # loss_result = loss.loss_for_batch(sim_out,gt_batch=YOLO_GT)
        #
        # print('loss_result size:',loss_result.size())


        # print(sim_out.size())
