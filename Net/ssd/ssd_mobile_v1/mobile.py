import torch.nn as nn
import torch
import torch.autograd
import numpy as np
import torchvision
from  Net.ssd.utils.visualize import get_img,show


import cv2
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1),#112,112
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),#56,56
            conv_dw(128, 256, 2),#28,28
            conv_dw(256, 256, 1),#28,28
            conv_dw(256, 512, 2),#14,14
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),#14,14
            conv_dw(512, 512, 1),#14,14
            conv_dw(512, 512, 1),#14,14
            conv_dw(512, 512, 1),#14,14
            conv_dw(512, 1024, 2),#7,7
            conv_dw(1024, 1024, 1),#7,7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__ == "__main__":

    model_name='/home/jason/下载/mobilenet_sgd_68.848.pth.tar'

    mobile_net = Net()

    state_dict=mobile_net.state_dict()
    save_dict=torch.load(model_name)
    for key in save_dict['state_dict']:
        print(key)

    for key in state_dict:

        save_key = 'module.'+key
        state_dict[key]=save_dict['state_dict'][save_key]
    mobile_net.load_state_dict(state_dict)

    mobile_net.cuda()
    mobile_net.eval()
    for i in range(100):
        x=torch.autograd.Variable(torch.rand([1,3,224,224]),volatile=True).cuda()

        t1=time.time()

        out = mobile_net(x)
        t2=time.time()

        print(out.size(),'time: ',1000*(t2-t1),' ms')

    # car_img_root='/home/jason/Dataset/CompCars/data/data/image/'
    # f=open('/home/jason/PycharmProjects/CUMT_YOLO/Dataset/test_car.txt','r')
    # car_list=f.readlines()
    # f.close()
    # car_list=[item.strip('\n') for item in car_list ]
    #
    # import PIL
    # import time
    #
    # test_trans = torchvision.transforms.Compose([
    #     # torchvision.transforms.CenterCrop((224,224)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    # ])
    #
    # f=open('/home/jason/PycharmProjects/CUMT_YOLO/Dataset/imagenet_label.txt','r')
    # label = f.readlines()
    # f.close()
    #
    # mobile_net.eval()
    # for i in range( car_list.__len__() ):
    #
    #     t1 = time.time()
    #
    #     car_img = PIL.Image.open(car_img_root+car_list[i])
    #
    #
    #     car_img = car_img.resize((224,224),PIL.Image.ANTIALIAS)
    #     car_img = test_trans(car_img)
    #     car_img=car_img.unsqueeze(0)
    #
    #     train_x = torch.autograd.Variable(car_img,volatile=True)
    #     outputs = mobile_net(train_x)
    #
    #     print(outputs.size())
    #     outputs=outputs.data.numpy()
    #
    #     result=np.argmax(outputs,axis=1)[0]
    #
    #     print(result,' : ',label[result])
    #
    #     img = get_img(car_img.numpy()[0])
    #
    #     cv2.imshow('img',img)
    #     cv2.waitKey(0)