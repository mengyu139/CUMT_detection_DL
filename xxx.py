import torch
import fire
import numpy as np
import torch.autograd


def calc(x):
    print('calc x*x= : ',str(int(x)*int(x)))
    # return x



def softmax(a,index):

    z=np.exp(a)
    r=np.exp(a[index])/z.sum()
    return r


a=np.ones([5,2])


b=np.array([0,0,1,0,0])

c=b==1
print(c,type(c))

print(a[c])

#
#
# #
# print(mask)
# print(mask.size())
#
# mask = mask.unsqueeze(1)
#
# print(mask.size())
#
# mask=mask.expand_as(a)
# print(mask.size())
#
# print(mask.float())
# # pos = a[pos_index]
# #
# # print(pos)

# a=torch.randn(8732, 4)
# a = torch.autograd.Variable(a)
#
# b=torch.ones(8732)
# b=b.type(torch.LongTensor)
# b = torch.autograd.Variable(b)
#
# conf_cost=torch.nn.functional.cross_entropy(input=a,target=b,size_average=False,reduce=False)
# conf_cost = conf_cost.data.numpy()
#
#
# print(conf_cost.size())