# coding=utf-8

import cv2
import torch
import xml.sax
import os

import Utils.xml2dict as xml2dict
import Utils.object_dict as Object_dict


# label_list=['person','car','bus','bicycle']

label_list=['car']

name_to_id={}
name_to_id['person']=0
name_to_id['car']=1
name_to_id['bus']=2
name_to_id['bicycle']=3


def make_dataset_txt(type,label_list,txt_source_root,xml_root,img_root):

    file_list=[]
    for i in range(label_list.__len__()):
        f = open(txt_source_root+label_list[i]+'_'+type+'.txt')
        lines=f.readlines()
        f.close()

        for item in lines:
            file_list.append(item.split(' ')[0])

    f=open(type+'.txt','w')

    file_list=set(file_list)

    for item in file_list:

        xml_file = xml_root+item+'.xml'

        print(xml_file)

        label = parse_a_xml(xml_file)

        if label['object'].__len__() > 0:
            f.write(item+' ')

            img = cv2.imread(img_root+item+'.jpg')
            f.write(str(img.shape[0])+' '+str(img.shape[1])+' ')

            f.write(str( label['object'].__len__() )+' ')

            for i in range( label['object'].__len__() ):

                f.write(str(label['object'][i][0])+' ')
                f.write(str(label['object'][i][1])+' ')
                f.write(str(label['object'][i][2])+' ')
                f.write(str(round(label['object'][i][3]/1./img.shape[1],3))+' ')
                f.write(str(round(label['object'][i][4]/1./img.shape[0],3))+' ')
                f.write(str(round(label['object'][i][5]/1./img.shape[1],3))+' ')
                f.write(str(round(label['object'][i][6]/1./img.shape[0],3))+' ')

            f.write('\n')

    f.close()


def parse_a_xml(xml_file):
    parser=xml2dict.XML2Dict()
    r=parser.parse(xml_file)

    label={}
    label['img_name']=xml_file.strip('.xml')+'.jpg'
    label['object']=[]

    # print('objects num : ',r.annotation.object.__len__())
    # print(type(r.annotation.object),isinstance(r.annotation.object,Object_dict.object_dict))

    if isinstance(r.annotation.object,Object_dict.object_dict):
        p =[]
        W=float(r.annotation.size.width)
        H=float(r.annotation.size.height)

        if r.annotation.object.name in label_list  :#and int(r.annotation.object.difficult) == 0
                # and (float(r.annotation.object.bndbox.xmax)-float(r.annotation.object.bndbox.xmin))/1./W * \
                # (float(r.annotation.object.bndbox.ymax)-float(r.annotation.object.bndbox.ymin))/1./H > 0.15*0.15:


            p.append(r.annotation.object.name)
            p.append(int(r.annotation.object.truncated))
            p.append(int(r.annotation.object.difficult))
            p.append(int(r.annotation.object.bndbox.xmin))
            p.append(int(r.annotation.object.bndbox.ymin))
            p.append(int(r.annotation.object.bndbox.xmax))
            p.append(int(r.annotation.object.bndbox.ymax))

            label['object'].append(p)
    else:
        for i in range( r.annotation.object.__len__()) :
            W=float(r.annotation.size.width)
            H=float(r.annotation.size.height)

            if r.annotation.object[i].name in label_list: #and int(r.annotation.object[i].difficult) == 0 \
                #     and  (float(r.annotation.object[i].bndbox.xmax)-float(r.annotation.object[i].bndbox.xmin))/1./W * \
                # (float(r.annotation.object[i].bndbox.ymax)-float(r.annotation.object[i].bndbox.ymin))/1./H > 0.15*0.15:

                p =[]
                p.append(r.annotation.object[i].name)
                p.append(int(r.annotation.object[i].truncated))
                p.append(int(r.annotation.object[i].difficult))
                p.append(int(r.annotation.object[i].bndbox.xmin))
                p.append(int(r.annotation.object[i].bndbox.ymin))
                p.append(int(r.annotation.object[i].bndbox.xmax))
                p.append(int(r.annotation.object[i].bndbox.ymax))

                label['object'].append(p)

    return label



def make_dataset_txt_V3(type,xml_root,img_root):

    xml_file_list = os.listdir(xml_root)
    xml_file_list = sorted(xml_file_list)

    f=open(type+'.txt','w')

    for item in xml_file_list:
        # print(item)
        xml_file = xml_root+item

        print(xml_file)
        label = parse_a_xml(xml_file)

        if label['object'].__len__() > 0:
            f.write(item.strip('.xml')+' ')

            img = cv2.imread(img_root+item.strip('.xml')+'.jpg')
            f.write(str(img.shape[0])+' '+str(img.shape[1])+' ')

            f.write(str( label['object'].__len__() )+' ')

            for i in range( label['object'].__len__() ):

                f.write(str(label['object'][i][0])+' ')
                f.write(str(label['object'][i][1])+' ')
                f.write(str(label['object'][i][2])+' ')
                f.write(str(round(label['object'][i][3]/1./img.shape[1],3))+' ')
                f.write(str(round(label['object'][i][4]/1./img.shape[0],3))+' ')
                f.write(str(round(label['object'][i][5]/1./img.shape[1],3))+' ')
                f.write(str(round(label['object'][i][6]/1./img.shape[0],3))+' ')

            f.write('\n')

    f.close()



if __name__=="__main__":
    img_root_train = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/JPEGImages/'
    img_root_test = '/home/jason/Dataset/VOCdevkit_2007_test/VOC2007/JPEGImages/'

    #
    # make_dataset_txt_V3('trainV3',
    #                  xml_root = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/Annotations/',
    #                  img_root=img_root_train
    #                  )
    # make_dataset_txt_V3('testV3',
    #              xml_root = '/home/jason/Dataset/VOCdevkit_2007_test/VOC2007/Annotations/',
    #              img_root=img_root_test
    #              )

    # make_dataset_txt('val',label_list=label_list,txt_source_root='/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/ImageSets/Main/',\
    #                  xml_root = '/home/jason/Dataset/VOCdevkit_2007_trainval/VOC2007/Annotations/',
    #                  img_root=img_root
    #                  )

    f = open('trainV3.txt','r')
    lines=f.readlines()
    lines=[item.strip('\n').split(' ') for item in lines]
    f.close()
    print(lines.__len__())
    #

    #
    for i in range(lines.__len__()):
        run_flag = True

        img = cv2.imread(img_root_train+lines[i][0]+'.jpg')
        for index in range( int(lines[i][3]) ):
            cv2.rectangle(img,pt1=( int(float(lines[i][7+index*7]) * float( img.shape[1]) ),\
                                    int(float(lines[i][8+index*7]) * float( img.shape[0]) )
                                    ),\
                              pt2=( int(float(lines[i][9+index*7]) * float( img.shape[1])),\
                                        int(float(lines[i][10+index*7]) * float( img.shape[0]))
                                        ),\
                          color=(0,255,0),
                          thickness=2
                          )

        while run_flag:
            key = cv2.waitKey(50)&0xff

            print(key)

            cv2.imshow('img',img)

            if key==110:
                run_flag=False
    #
    #
    #
    #
    # #
    # #
    # # label = parse_a_xml("004584.xml")
    # #
    # # print(label)
    # #
    # # img = cv2.imread(img_root+label['img_name'])
    # #
    # # for i in range( label['object'].__len__() ):
    # #
    # #     cv2.rectangle(img,pt1=(label['object'][i][3],label['object'][i][4]),\
    # #                   pt2=(label['object'][i][5],label['object'][i][6]),\
    # #                   color=(0,255,0),thickness=2)
    # #
    # # cv2.imshow('img',img)
    # # cv2.waitKey(0)
    # #
    # #
    # # print(label)
