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
import Net.ssd.loss.loss as Loss

sys.path.append('/home/jason/PycharmProjects/CUMT_YOLO/Net/ssd_vgg')

from Net.ssd.layers.modules.l2norm import L2Norm


# out_locs:[b,8732,4] Variable
# out_confs:[b,8732,num_classes=2] Variable
# priors : [8732,4] Variable
# Return-> boxes array [b,4]  [x1 y1 x2 y2] 0.0~1.0  valid [b]
def get_best_prior_box(out_locs,out_confs,priors):

    result_boxs=np.zeros([out_locs.size(0),4])
    valid=[]

    for i in range(out_locs.size(0)):
        loc_boxs=Loss.decode(out_locs[i].cpu().data, priors=priors.cpu().data, variances=[0.1, 0.2])
        loc_boxs=loc_boxs.numpy()

        conf_result = torch.nn.functional.softmax(out_confs[i],dim=1)
        conf_result = conf_result.data.cpu().numpy()

        index = np.argmax(conf_result,axis=1)

        car_index = index==1

        if car_index.sum()==0:
            valid.append(0)
            continue

        loc_boxs=loc_boxs[car_index]
        conf_result=conf_result[car_index]

        max_index=np.argmax(conf_result[:,1])

        result_boxs[i,...]=loc_boxs[max_index,...]

        if loc_boxs[max_index,2]>loc_boxs[max_index,0] and loc_boxs[max_index,3]>loc_boxs[max_index,1]:
            valid.append(1)
        else:
            valid.append(0)

    result_boxs=np.clip(result_boxs,a_max=1.0,a_min=0)
    return result_boxs,valid


# input_features:[b,512,19,19]      Variable

# best_boxs[0] array [b,4]
# best_boxs[1] list [b]

# index:  number
# extracted_features :[b,512,1,1]   Variable
def extract_features(input_features,best_boxs,index,extracted_features,img_size=300):

    box=best_boxs[0][index]
    valid=best_boxs[1][index]

    if valid == 0:
        return
    else:

        ceil_num=input_features.size(2)

        step=img_size/1./ceil_num
        box=box*img_size
        box=np.clip(box,a_max=img_size,a_min=0)

        box=box/1./step
        box=np.clip(box,a_min=0,a_max=ceil_num-1)
        box=np.floor(box)
        box=box.astype(np.int)# grid [0~19-1]

        m=input_features[index][:,box[1]:box[3]+1,box[0]:box[2]+1]
        m=torch.max(m,dim=1, keepdim=True)[0]
        m=torch.max(m,dim=2, keepdim=True)[0]

        extracted_features[index,...]=m[...]



class SSD_Net(nn.Module):

    def __init__(self, base,extras=None,head=None,num_classes=2,fg_classes=19):
        super(SSD_Net, self).__init__()

        self.vgg = torch.nn.ModuleList(base)

        self.L2Norm = L2Norm(512,20)

        self.extras = nn.ModuleList(extras)

        self.L2Norm = L2Norm(512, 20)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.num_classes=num_classes

        self.priorbox = PriorBox(v2)
        self.priors =  torch.autograd.Variable(self.priorbox.forward(),requires_grad=False)
        self.size = 300

        self.fg_classes=fg_classes


        self.fg_conv=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fg_classifier=nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.fg_classes),
        )
        self._initialize_weights()


    def forward(self, x):

        sources=[]

        # vgg conv 4_3 [38]
        for k in range(23):
            x = self.vgg[k](x)
            FG_conv_feature=x

        s = self.L2Norm(x)


        # vgg conv 4_3
        sources.append(s)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            # print('k: ',k)
            # if k == 29:
            #     FG_conv_feature=x
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
        loc=loc.view(loc.size(0), -1, 4)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf],1)
        conf=conf.view(conf.size(0), -1, self.num_classes)

        # ----------FG---------------
        FG_conv_feature=self.fg_conv(FG_conv_feature)
        Extracted_features=torch.autograd.Variable(torch.zeros((loc.size(0),512,1,1))).cuda()
        best_prior=get_best_prior_box(loc,conf,self.priors)

        for index in range(loc.size(0)):
            extract_features(FG_conv_feature,best_prior,index=index,extracted_features=Extracted_features,img_size=300)

        Extracted_features=Extracted_features.view(Extracted_features.size(0),-1)

        # print('extract_features size: ',Extracted_features.size())

        fg_cls=self.fg_classifier(Extracted_features)

        # =========================

        output = (
                    loc,
                    conf,
                    # loc.view(loc.size(0), -1, 4),
                    # conf.view(conf.size(0), -1, self.num_classes),
                    # FG_conv_feature,
                    fg_cls,
                    best_prior # []
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
    net.cuda()

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
    data=torch.autograd.Variable(torch.FloatTensor(1,3,300,300)).cuda()

    out = net(data)

    print('out size',out[2].size())
    print('fg cls size',out[3].size())

    # for item in out:
    #     print(item.size())

