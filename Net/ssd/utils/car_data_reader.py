# coding=utf-8

import torch.nn
import torch.nn.functional
import torch.autograd
import torch.tensor
import torch.optim
import torchvision
import torch.utils.data
import torchvision.transforms

from visdom import Visdom
# from graphviz import Digraph

import math
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image,PIL.ImageEnhance
import os
import time
import json
import collections
import random
import cv2



class CustomDataset(torch.utils.data.Dataset):#需要继承data.Dataset

    def __init__(self,images_root,txt_path,is_train,img_size=300):
        self.image_root = images_root
        self.is_train = is_train
        self.img_size=img_size


        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize ((self.img_size,self.img_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])


        f = open( txt_path ,'r')
        self.lines = f.readlines()
        self.lines=[item.strip('\n').split(' ') for item in self.lines]
        f.close()


    def __getitem__(self, index):

        image_path = self.image_root+self.lines[index][0]
        img = PIL.Image.open(image_path)
        flip=random.randint(0,100)%2

        if flip==1:
            img = img.transpose( PIL.Image.FLIP_LEFT_RIGHT )

        # img = img.resize((self.img_size,self.img_size),PIL.Image.ANTIALIAS)

        if self.is_train:
            a = random.uniform(0.75,1.25)
            img = PIL.ImageEnhance.Color(img).enhance(a)

            a = random.uniform(0.75,1.25)
            img = PIL.ImageEnhance.Brightness(img).enhance(a)

            a = random.uniform(0.75,1.25)
            img = PIL.ImageEnhance.Contrast(img).enhance(a)

        img = self.transform(img)

        img_name = self.lines[index][0]

        return img,index,img_name,flip

    def __len__(self):
        return len( self.lines )


    def get_GT_boxs(self,indexs,flips):

        batch_GT_boxs=[]
        for cnt,index  in  enumerate(indexs) :
            # print( 'file: ', self.lines[index][0],' pbj num: ',self.lines[index][3])

            img_GT_boxs=[]
            for i in range( 1 ):
                #x1 y1 x2 y2 class fg_class
                box=[0,0,0,0,0,0]
                b=[0,0,0,0,0,0]

                b[0]=float(self.lines[index][2])
                b[1]=float(self.lines[index][3])
                b[2]=float(self.lines[index][4])
                b[3]=float(self.lines[index][5])
                b[4]=0# for detecttion
                b[5]=int(self.lines[index][1])

                if flips[cnt]:
                    box[0]=round(1.0-b[2],3)
                    box[1]=round(b[1],3)
                    box[2]=round(1.0-b[0],3)
                    box[3]=round(b[3],3)
                    box[4]=int(b[4])
                    box[5]=int(b[5])
                else:
                    box[0]=round(b[0],3)
                    box[1]=round(b[1],3)
                    box[2]=round(b[2],3)
                    box[3]=round(b[3],3)
                    box[4]=int(b[4])
                    box[5]=int(b[5])

                img_GT_boxs.append(box)
            batch_GT_boxs.append(img_GT_boxs)
        return batch_GT_boxs


    def get_SSD_GTS(self,batch_GT_boxs):
        # x y w h class

        YOLO_GTS=[]
        for b in range( batch_GT_boxs.__len__() ):
            gt_for_one_img=[]
            for index,box in enumerate( batch_GT_boxs[b] ):
                # x y w h cls fg_cls
                gt=[0,0,0,0,0,0]
                gt[0]=(box[0]+box[2])/2
                gt[1]=(box[1]+box[3])/2
                gt[2]=box[2]-box[0]
                gt[3]=box[3]-box[1]
                gt[4]=box[4]#class
                gt[5]=box[5]#fg_cls

                gt_for_one_img.append(gt)

            YOLO_GTS.append(gt_for_one_img)
        return YOLO_GTS






if __name__=="__main__":

    pass