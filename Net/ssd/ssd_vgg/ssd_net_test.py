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





if __name__ =="__main__":

    # =============Set up the configuration==========================================
    train_img_root='/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'
    test_img_root='/home/jason/Dataset/VOCdevkit_2007_test/VOC2007/JPEGImages/'
    train_btach_size=1

    num_classes=2#  background:0  car:1

    Use_cuda = True
    Use_vis = False
    Use_imshow=True
    save_mode_name = './ssd_net.pth'


    # =============Set up the net==========================================
    vgg_layers = make_vgg_layers(cfg['D'],batch_norm=False)
    extra_layers=make_ectra_layers(extras_cfg['300'],1024,False)
    head = make_loc_conf_layers(input_features_list=[512,1024,512,256,256,256],box_cfg=box_cfg,num_classes=num_classes)
    ssd_net = SSD_Net(vgg_layers,extras=extra_layers,head=head,num_classes=2)

    # =============Load the parameters==========================================

    save_dict = torch.load( save_mode_name )
    ssd_net.load_state_dict(save_dict)


    # =============Set the data loader==========================================
    train_dataset = CustomDataset(test_img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/testV3.txt',\
                                        is_train=False,label_map={'car':0},
                                        img_size=300)
    # test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
    data_loader={}
    data_loader["train"]=torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=train_btach_size,
                                                     shuffle=False ,
                                                     num_workers=8)
    # data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)


    # =============Start the train==========================================
    if Use_cuda:
        ssd_net.cuda()
    else:
        ssd_net.cpu()

    for epoch in range(1):

        GPU_TEM = read_temperature()
        print ('+++++++++++++++++++++++++++++++++++++gpu tem :',GPU_TEM)
        if GPU_TEM > 80:
            print ('GPU OVER HEATED,QUIT!!!')
            break


        ssd_net.eval()

        COST=0
        COST_CNT=0

        TP=0
        FP=0
        FN=0

        s_1=[]
        s_2=[]
        s_pickle=open('save.pickle','wb')

        for item in data_loader['train']:

            car_img,indexs,img_names,flips = item
            Ori_GTS=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
            GTS=train_dataset.get_SSD_GTS(Ori_GTS)

            train_x = torch.autograd.Variable(car_img,volatile=True).cuda()
            outputs = ssd_net(train_x)

            # [x1 y1 x2 y2] tensor  [8732,4] -> ndarray
            loc_boxs=Loss.decode(outputs[0][0].cpu().data, priors=ssd_net.priors.data, variances=[0.1, 0.2])
            loc_boxs=loc_boxs.numpy()

            # tensor [8732,2] -> ndarray
            conf_result = outputs[1][0]
            conf_result = torch.nn.functional.softmax(conf_result,dim=1)
            conf_result = conf_result.data.cpu().numpy()

            print(loc_boxs.shape,conf_result.shape)

            index = np.argmax(conf_result,axis=1)

            car_index = index==1

            loc_boxs=loc_boxs[car_index]
            conf_result=conf_result[car_index][:,1:2]

            # print('loc_boxs size: ',loc_boxs.shape,' conf_result size: ',conf_result.shape)

            result=np.concatenate((loc_boxs,conf_result),axis=1)

            result=nms(result,threshold=0.5)

            s_1.append(result)
            s_2.append(Ori_GTS[0])


            # tp,fp,fn=eval_img(result=result,gt=np.array(Ori_GTS[0]),threshold=0.5)
            #
            # print("tp is %d ,fp is %d ,fn is %d gt is %d " %(tp,fp,fn,Ori_GTS[0].__len__()))
            #
            #
            if Use_imshow:
                img = get_img(car_img.numpy()[0])
                if result is not None:
                    for i in range(result.shape[0]):
                        pt=result[i]*300
                        cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,255,0),
                                      thickness=2
                                      )

                gt_box=np.array(Ori_GTS[0])
                for i in range(gt_box.shape[0]):
                        pt=gt_box[i]*300
                        cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,0,255),
                                      thickness=1
                                      )


                cv2.imshow('img',img)

                cv2.waitKey(0)
        # pickle.dump([s_1,s_2],s_pickle)
        # s_pickle.close()





        # for item in data_loader['train']:
        #     COST_CNT+=1
        #
        #     imgs,indexs,img_names,flips = item
        #     GTS=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
        #     GTS=train_dataset.get_SSD_GTS(GTS)
        #
        #     if Use_cuda:
        #         train_x = torch.autograd.Variable(imgs,volatile=True).cuda()
        #     else:
        #         train_x = torch.autograd.Variable(imgs,volatile=True)
        #
        #     outputs = ssd_net(train_x)
        #
        #     loss = Loss.loss_for_batch(ssd_net.priors,
        #                                GTS=GTS,
        #                                outputs=outputs,
        #                                threshold=0.5,k=3,is_cuda=Use_cuda
        #                                )
        #
        #     # [x1 y1 x2 y2] tensor  [8732,4] -> ndarray
        #     loc_boxs=Loss.decode(outputs[0][0].cpu().data, priors=ssd_net.priors.data, variances=[0.1, 0.2])
        #     loc_boxs=loc_boxs.numpy()
        #
        #
        #
        #     # tensor [8732,2] -> ndarray
        #     conf_result = outputs[1][0]
        #     conf_result = torch.nn.functional.softmax(conf_result,dim=1)
        #     conf_result = conf_result.data.cpu().numpy()
        #
        #     print(loc_boxs.shape,conf_result.shape)
        #
        #     index = np.argmax(conf_result,axis=1)
        #
        #     car_index = index==1
        #
        #     loc_boxs=loc_boxs[car_index]
        #     conf_result=conf_result[car_index]
        #
        #     img = get_img(imgs.numpy()[0])
        #
        #
        #     for i in range(loc_boxs.shape[0]):
        #         pt=loc_boxs[i]*300
        #         cv2.rectangle(img,pt1=( int(pt[0]),int(pt[1]) ),pt2=( int(pt[2]),int(pt[3]) ),color=(0,255,0),
        #                       thickness=2
        #                       )
        #     cv2.imshow('img',img)
        #
        #     cv2.waitKey(0)

