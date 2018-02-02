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

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000,default_box_num=1):
        super(VGG, self).__init__()

        self.num_classes=num_classes

        self.features = features

        self.extra_conv=nn.Sequential(
            torch.nn.Conv2d(512,512,3,stride=1,padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(512,512,3,stride=1,padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(512,512,3,stride=1,padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True),

            torch.nn.AvgPool2d(kernel_size=2,stride=2)
        )

        self.fc_layer= nn.Sequential(

            nn.Linear(7*7*512, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 7*7*(num_classes+5*2)),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def forward(self, data):

        conv_feature_1 = self.features(data)
        conv_feature_2 = self.extra_conv(conv_feature_1)

        flatern_feature=conv_feature_2.view(conv_feature_2.size(0),-1)

        result = self.fc_layer(flatern_feature)

        result = result.view(result.size(0),self.num_classes+5*2,7,7)

        # very dirty code ... try to add offset to output,since our output is relative to the grid left-up corner[0.0~1.0]
        # ----- -----------------------------------------------------------------------------
        a=np.array([0,1,2,3,4,5,6])
        a = np.reshape(a,[1,7])
        a= np.tile(a,[7,1])
        a = a[np.newaxis,...]
        a=np.tile(a,[2,1,1])

        # a = torch.autograd.Variable( torch.FloatTensor(a) ).cuda()

        b=np.array([0,1,2,3,4,5,6])
        b = np.reshape(b,[7,1])
        b= np.tile(b,[1,7])
        b = b[np.newaxis,...]
        b=np.tile(b,[2,1,1])
        # b = torch.autograd.Variable( torch.FloatTensor(b) ).cuda()

        x_offset=np.zeros([result.size(0),14,7,7])
        x_offset[:,4:6,...]=a
        x_offset[:,6:8,...]=b
        x_offset = torch.autograd.Variable( torch.FloatTensor(x_offset) ).cuda()

        result = result + x_offset

        mask=np.ones([result.size(0),14,7,7])
        mask[:,4:8,...]=7.
        mask = torch.autograd.Variable( torch.FloatTensor(mask) ).cuda()

        result = result / mask

        # # add offset to x x y y
        # result[:,4:6] = result[:,4:6]+a
        # result[:,6:8] = result[:,6:8]+b
        #
        # #covert 0~7 to 0~1.0 reletive to the whole img size
        # result[:,4:6] = result[:,4:6]/7.
        # result[:,6:8] = result[:,6:8]/7.
        # # -------------------------------------------------------------------------

        return conv_feature_1,conv_feature_2,result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(0.1, inplace=True),]
            else:
                layers += [conv2d, nn.LeakyReLU(0.1, inplace=True),]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'MY': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
