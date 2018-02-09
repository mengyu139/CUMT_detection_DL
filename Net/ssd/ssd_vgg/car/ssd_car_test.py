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
import cv2
from visdom import Visdom

from Net.ssd.utils.car_data_reader import CustomDataset
from  Net.ssd.utils.visualize import get_img,show
from  Net.ssd.ssd_vgg.car.car_base import SSD_Net,make_vgg_layers,make_ectra_layers,make_loc_conf_layers,cfg,box_cfg,extras_cfg
from Net.ssd.loss import loss as Loss
from Net.ssd.utils.read_tmp import read_temperature
from Net.ssd.utils.visualize import IOU



if __name__ =="__main__":

    # =============Set up the configuration==========================================
    train_img_root='/home/jason/Dataset/CompCars/data/data/image/'
    train_btach_size=1

    num_classes=2#  background:0  car:1

    Use_cuda = True
    Use_vis = False
    save_mode_name = 'ssd_cumt_car_fg.pth'

    # =============Set up the net==========================================
    vgg_layers = make_vgg_layers(cfg['D'],batch_norm=False)
    extra_layers=make_ectra_layers(extras_cfg['300'],1024,False)
    head = make_loc_conf_layers(input_features_list=[512,1024,512,256,256,256],
                                box_cfg=box_cfg,
                                num_classes=num_classes)
    ssd_net = SSD_Net(vgg_layers,extras=extra_layers,head=head,num_classes=2)

    # =============Load the parameters==========================================
    save_dict = torch.load( save_mode_name )
    ssd_net.load_state_dict(save_dict)

    # save_dict = torch.load( save_mode_name )
    # state_dict=ssd_net.state_dict()
    # for key in save_dict:
    #     state_dict[key]=save_dict[key]
    # ssd_net.load_state_dict(state_dict)

    # save_dict=torch.load('/home/jason/PycharmProjects/CUMT_YOLO/Model/vgg16-397923af.pth')
    # state_dict=ssd_net.state_dict()
    # for key in save_dict:
    #     if 'features.' in key:
    #         target_key = 'vgg.'+key.lstrip('features.')
    #         print(target_key)
    #
    #         state_dict[target_key]=save_dict[key]
    #
    # ssd_net.load_state_dict(state_dict)
    # =============Set the optimizer==========================================
    # k=0.0001
    # optimizer = torch.optim.SGD ([
    #     # {'params':train_para ,'lr':0.001},
    #     {'params':ssd_net.vgg.parameters(),'lr':k},
    #     {'params':ssd_net.extras.parameters(),'lr':k},
    #     {'params':ssd_net.loc.parameters(),'lr':k},
    #     {'params':ssd_net.conf.parameters(),'lr':k},
    # ],weight_decay=0.0,momentum=0.9)

    # =============Set the data loader==========================================
    train_dataset = CustomDataset(train_img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Net/ssd/data/car_test.txt',
                                        is_train=False,
                                        img_size=300)
    # test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
    data_loader={}
    data_loader["train"]=torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=train_btach_size,
                                                     shuffle=False,
                                                     num_workers=1)
    # data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)


     # =============Start the train==========================================
    if Use_cuda:
        ssd_net.cuda()
    else:
        ssd_net.cpu()


    ssd_net.eval()

    cnt=0
    correct_cnt=0
    for item in data_loader['train']:

        imgs,indexs,img_names,flips = item
        GTS=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
        GTS=train_dataset.get_SSD_GTS(GTS)

        # print(imgs.size())
        # img = get_img(imgs.numpy()[0])
        # show(img,gts=GTS[0])
        #
        # cv2.waitKey(0)

        if Use_cuda:
            train_x = torch.autograd.Variable(imgs,volatile=True).cuda()
        else:
            train_x = torch.autograd.Variable(imgs,volatile=True)

        outputs = ssd_net(train_x)

        img = get_img(imgs.numpy()[0])
        pt=outputs[3][0][0]*300
        # pt=loc_boxs[max_index]*300


        cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,255,0),
                      thickness=2
        )

        gt=GTS[0][0][0:4]
        gt_box=np.array([0,0,0,0],dtype=np.float)
        gt_box[0]=gt[0]-0.5*gt[2]
        gt_box[1]=gt[1]-0.5*gt[3]
        gt_box[2]=gt[0]+0.5*gt[2]
        gt_box[3]=gt[1]+0.5*gt[3]

        print('iou:',IOU( outputs[3][0][0], gt_box))

        fg_pred=outputs[2][0].cpu().data.numpy()
        fg_cls=np.argmax(fg_pred)
        print('real label:',GTS[0][0][5],' predict cls: ',fg_cls)

        cnt+=1
        if GTS[0][0][5] == fg_cls:
            correct_cnt+=1


        pt = gt_box * 300
        cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(255,0,0),
                      thickness=2
        )

        # for i in range(loc_boxs.shape[0]):
        #     pt=loc_boxs[i]*300
        #     cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,255,0),
        #                   thickness=2
        #                   )

        cv2.imshow('img',img)
        cv2.waitKey(0)

    print('cnt: ',cnt,' correct_cnt: ',correct_cnt,' accuracy: ',correct_cnt/1./cnt*100,"%")