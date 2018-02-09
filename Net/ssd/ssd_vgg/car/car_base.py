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
import torch.nn.functional as F
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

    out_size=extracted_features.size(2)

    if valid == 0:
        return
    else:

        ceil_num=input_features.size(2)

        step=img_size/1./ceil_num
        box=box*img_size
        box=np.clip(box,a_max=img_size,a_min=0)

        box=box/1./step
        box[0:2]=box[0:2]-1
        box[2:4]=box[2:4]+1

        box=np.clip(box,a_min=0,a_max=ceil_num-1)
        box=np.floor(box)
        box=box.astype(np.int)# grid [0~19-1]

        m=torch.nn.functional.adaptive_max_pool2d(input_features[index:index+1,:,box[1]:box[3]+1,box[0]:box[2]+1],
                                                  output_size=(out_size,out_size))

        extracted_features[index]=m.squeeze(0)


        # print('a extracted_features size: ',extracted_features[index,...].size())



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

        self.fg_max_pool_size=15

        self.fg_classes=fg_classes


        # self.fg_conv=nn.Sequential(
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        #
        # )

        self.fg_conv=nn.Sequential(
            InceptionD(512),
            InceptionE(1024),
            InceptionE(2048),
            nn.AvgPool2d(kernel_size=7),
            nn.Dropout(0.5)
        )


        self.fg_classifier=nn.Sequential(
            # nn.Linear(2048, 4096),
            # nn.ReLU(True),
            # nn.Dropout(0.8),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(0.75),
            nn.Linear(2048, self.fg_classes),
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
        #
        Extracted_features=torch.autograd.Variable(torch.zeros((loc.size(0),512,self.fg_max_pool_size,self.fg_max_pool_size))).cuda()
        best_prior=get_best_prior_box(loc,conf,self.priors)

        for index in range(loc.size(0)):
            extract_features(FG_conv_feature,best_prior,index=index,extracted_features=Extracted_features,img_size=300)

        FG_conv_feature=self.fg_conv(Extracted_features)

        # print('FG_conv_feature size: ',FG_conv_feature.size())

        FG_conv_feature=FG_conv_feature.view(FG_conv_feature.size(0),-1)

        # print('FG_conv_feature size: ',FG_conv_feature.size())

        fg_cls=self.fg_classifier(FG_conv_feature)

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



class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3,branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


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

# --------------Inception----------------------------------------
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)




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

