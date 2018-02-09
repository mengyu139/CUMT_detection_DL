import torch
import fire
import numpy as np
import torch.autograd
import torch.nn.functional

def calc(x):
    print('calc x*x= : ',str(int(x)*int(x)))
    # return x



def softmax(a,index):

    z=np.exp(a)
    r=np.exp(a[index])/z.sum()
    return r


import torchvision
torchvision.models.inception_v3()


# save_dict=torch.load('/home/jason/PycharmProjects/CUMT_YOLO/Model/vgg16-397923af.pth')
#
# for key in save_dict:
#     print(key)

print(torch.backends.cudnn.version())

# a=torch.autograd.Variable(torch.FloatTensor(a))
# b=torch.autograd.Variable(torch.LongTensor(b))
#
# s = torch.nn.functional.cross_entropy(input=a,target=b,reduce=False,size_average=True)
# print(s)
