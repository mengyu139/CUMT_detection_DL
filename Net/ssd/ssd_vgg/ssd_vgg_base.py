# coding=utf-8

import torch.nn as nn
import torch.nn.functional
import torch.autograd
import torch.tensor
import torch.optim
import torchvision
import torch.utils.data
import torchvision.transforms
import torch.nn.init

# torch.nn.init.xavier_normal()

import math
import numpy as np
import sys

from Net.ssd.data.config import v2
from Net.ssd.layers.modules.prior_box import PriorBox

sys.path.append('/home/jason/PycharmProjects/CUMT_YOLO/Net/ssd_vgg')

from Net.ssd.layers.modules.l2norm import L2Norm


class SSD_Net(nn.Module):

    def __init__(self, base,extras=None,head=None,num_classes=2,fg_classes=19):
        super(SSD_Net, self).__init__()

        self.vgg = torch.nn.ModuleList(base)

        self.L2Norm = L2Norm(512,20)

        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.num_classes=num_classes

        self.priorbox = PriorBox(v2)
        self.priors =  torch.autograd.Variable(self.priorbox.forward(),requires_grad=False)
        self.size = 300

        self.fg_classes=fg_classes

        # for m in self.modules():
        #     print(type(m),m.__class__.__name__)

        self._initialize_weights()

        # for index,item in enumerate(self.vgg) :
            # print(index," : " ,item.__class__.__name__ )

        # print('self.loc: ',self.loc.__len__())

        # self.fg_classifier=nn.Sequential(
        #
        #     nn.Linear(512, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, self.fg_classes),
        # )

    def forward(self, x):

        sources=[]

        # vgg conv 4_3
        for k in range(23):
            x = self.vgg[k](x)


        s = self.L2Norm(x)
        # vgg conv 4_3
        sources.append(s)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            # print('k: ',k)
            if k == 29:
                conv_feature=x
                # print('conv_feature size: ',conv_feature.size())


        # vgg fc7->cov7
        sources.append(x)

        # extra convs
        for k, v in enumerate(self.extras):
            x = torch.nn.functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        loc=[]
        conf=[]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #换顺序，从[b,c,h,w]->[b,h,w,c]
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # [b,xxx]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf],1)

        output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    conv_feature
            )

        return output


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal (m.weight.data)

                if m.bias is not None:
                     m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_vgg_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU( inplace=True),]
            else:
                layers += [conv2d, nn.ReLU( inplace=True),]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)            # out [512 19 19]
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  #[1024 19 19]
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)                        #[1024 19 19]

    layers += [pool5,conv6,conv7]
    return layers



def make_ectra_layers(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
                # print( 'conv2d ',' in:',in_channels,' out:',cfg[k + 1],' kernel:',(1, 3)[flag],' stride:',2,' padding:',1 )
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                # print( 'conv2d ','in: ',in_channels,'out: ',v,'kernel: ',(1, 3)[flag],'stride: ',1,'padding: ',0 )

            flag = not flag
        in_channels = v
    return layers


def make_loc_conf_layers(input_features_list,box_cfg,num_classes=1+1):
    loc_layers = []
    conf_layers = []


    for i,item in enumerate(input_features_list) :
        loc_layers += [torch.nn.Conv2d(input_features_list[i],box_cfg[i]*4,kernel_size=3,padding=1)]
        conf_layers += [torch.nn.Conv2d(input_features_list[i],box_cfg[i]*num_classes,kernel_size=3,padding=1)]

    return loc_layers,conf_layers







box_cfg=[4,6,6,6,4,4]
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],#就是你了，VGG16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'MY': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

extras_cfg = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}





if __name__=="__main__":


    # make_ectra_layers(extras_cfg['300'],1024,False)
    vgg_layers = make_vgg_layers(cfg['D'],batch_norm=False)
    extra_layers=make_ectra_layers(extras_cfg['300'],1024,False)
    head = make_loc_conf_layers(input_features_list=[512,1024,512,256,256,256],box_cfg=box_cfg,num_classes=2)

    net = SSD_Net(vgg_layers,extras=extra_layers,head=head,num_classes=2)


    # state_dict = net.state_dict()
    #
    # # for key in state_dict:
    # #     print(key)
    #
    # model_name='/home/jason/PycharmProjects/CUMT_YOLO/Model/vgg16-397923af.pth'
    # save_dict = torch.load(model_name)
    #
    # cnt=-1
    # for key in save_dict:
    #     cnt+=1
    #     print(cnt,' ',key)
    #
    #
    data=torch.autograd.Variable(torch.FloatTensor(1,3,300,300))

    out = net(data)

    print('out size',out[2].size())


    # for item in out:
    #     print(item.size())

