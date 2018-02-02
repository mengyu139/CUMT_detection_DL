import torch
import fire
import numpy as np
import torch.autograd


def calc(x):
    print('calc x*x= : ',str(int(x)*int(x)))
    # return x

a=[[1,2],[1,2],[3,4],[7,6]]

a=np.array(a)

x=np.argmax(a,axis=1)

y =x==1

b=a[y]

print(b)



a
s
for i in range(10):
    s+= a*a


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