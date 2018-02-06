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




a=torch.autograd.Variable(torch.FloatTensor())

b=a>1

print(b,)


# a=[[1,2,3],[1,2,3],[1,2,3]]
# b=[1,1,1]
# a=torch.autograd.Variable(torch.FloatTensor(a))
# b=torch.autograd.Variable(torch.LongTensor(b))
#
# s = torch.nn.functional.cross_entropy(input=a,target=b,reduce=False,size_average=True)
# print(s)
