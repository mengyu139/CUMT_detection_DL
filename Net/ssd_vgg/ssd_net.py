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

from Net.ssd_vgg.utils.data_read import CustomDataset
from  Net.ssd_vgg.utils.visualize import get_img,show
from  Net.ssd_vgg.ssd_vgg_base import SSD_Net,make_vgg_layers,make_ectra_layers,make_loc_conf_layers,cfg,box_cfg,extras_cfg
from Net.ssd_vgg import loss as Loss
from Net.ssd_vgg.utils.read_tmp import read_temperature



if __name__ =="__main__":

    # =============Set up the configuration==========================================
    train_img_root='/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'
    train_btach_size=16

    num_classes=2#  background:0  car:1

    Use_cuda = True
    Use_vis = True
    save_mode_name = 'ssd_net.pth'

    # =============Set up the net==========================================
    vgg_layers = make_vgg_layers(cfg['D'],batch_norm=False)
    extra_layers=make_ectra_layers(extras_cfg['300'],1024,False)
    head = make_loc_conf_layers(input_features_list=[512,1024,512,256,256,256],box_cfg=box_cfg,num_classes=num_classes)
    ssd_net = SSD_Net(vgg_layers,extras=extra_layers,head=head,num_classes=2)

    # =============Load the parameters==========================================

    save_dict = torch.load( save_mode_name )
    ssd_net.load_state_dict(save_dict)


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
    k=0.0002
    optimizer = torch.optim.SGD ([
        # {'params':train_para ,'lr':0.001},
        {'params':ssd_net.vgg.parameters(),'lr':k},
        {'params':ssd_net.extras.parameters(),'lr':k},
        {'params':ssd_net.loc.parameters(),'lr':k},
        {'params':ssd_net.conf.parameters(),'lr':k},
    ],weight_decay=0.0,momentum=0.9)

    # =============Set the data loader==========================================
    train_dataset = CustomDataset(train_img_root,txt_path='/home/jason/PycharmProjects/CUMT_YOLO/Dataset/trainV3.txt',\
                                        is_train=True,label_map={'car':0},
                                        img_size=300)
    # test_dataset = data_read.CustomDataset(img_root,txt_path='new_26_data_test.txt',is_train=False,label_map_txt='26_label.txt')
    data_loader={}
    data_loader["train"]=torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=train_btach_size,
                                                     shuffle=True,
                                                     num_workers=8)
    # data_loader["test"]=torch.utils.data.DataLoader(test_dataset, batch_size=test_btach_size,shuffle=False, num_workers=8)


    # =============Set the visdom==========================================

    viz = Visdom()
    # line updates
    Vis_loss = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
         opts=dict(
                xlabel='step',
                ylabel='Loss',
                title='step SSD Training Loss',
                # legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
    )
    Vis_epoch_loss = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
         opts=dict(
                xlabel='epoch',
                ylabel='Loss',
                title='epoch SSD Training Loss',
                # legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
    )


    dis_cnt=0


    # =============Start the train==========================================
    if Use_cuda:
        ssd_net.cuda()
    else:
        ssd_net.cpu()

    for epoch in range(1000):

        GPU_TEM = read_temperature()
        print ('+++++++++++++++++++++++++++++++++++++gpu tem :',GPU_TEM)
        if GPU_TEM > 80:
            print ('GPU OVER HEATED,QUIT!!!')
            break

        if epoch < 100:
            k=0.0002
            optimizer = torch.optim.SGD ([
                # {'params':train_para ,'lr':0.001},
                {'params':ssd_net.vgg.parameters(),'lr':k},
                {'params':ssd_net.extras.parameters(),'lr':k},
                {'params':ssd_net.loc.parameters(),'lr':k},
                {'params':ssd_net.conf.parameters(),'lr':k},
            ],weight_decay=0.0,momentum=0.9)

        else:
            k=0.0001
            optimizer = torch.optim.SGD ([
                # {'params':train_para ,'lr':0.001},
                {'params':ssd_net.vgg.parameters(),'lr':k},
                {'params':ssd_net.extras.parameters(),'lr':k},
                {'params':ssd_net.loc.parameters(),'lr':k},
                {'params':ssd_net.conf.parameters(),'lr':k},
            ],weight_decay=0.0,momentum=0.9)



        ssd_net.train()

        COST=0
        COST_CNT=0

        for item in data_loader['train']:
            dis_cnt+=1
            COST_CNT+=1

            imgs,indexs,img_names,flips = item
            GTS=train_dataset.get_GT_boxs(indexs.numpy(),flips.numpy())
            GTS=train_dataset.get_SSD_GTS(GTS)

            if Use_cuda:
                train_x = torch.autograd.Variable(imgs).cuda()
            else:
                train_x = torch.autograd.Variable(imgs)

            outputs = ssd_net(train_x)

            optimizer.zero_grad()


            loss = Loss.loss_for_batch(ssd_net.priors,
                                       GTS=GTS,
                                       outputs=outputs,
                                       threshold=0.5,k=3,is_cuda=Use_cuda
                                       )

            loss.backward()
            optimizer.step()


            # print(loss.cpu().data.numpy()[0] , loss.requires_grad)

            viz.line(
            X=np.array([dis_cnt]),
            Y=np.array( [loss.cpu().data.numpy()[0]] ), win=Vis_loss, update='append')

            COST+=loss.cpu().data.numpy()[0]

        print('loss for one epoch :',COST/1./COST_CNT)
        viz.line(
            X=np.array([epoch]),
            Y=np.array( [COST] ), win=Vis_epoch_loss, update='append')



        if epoch % 10 == 0:
            torch.save(ssd_net.state_dict(),save_mode_name)
            print('----------------------save modle in ',save_mode_name )