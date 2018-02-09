import numpy as np

import pickle

# x={}
# x['a']=np.zeros([10,10])
# x['b']=np.ones([20,20])
#
# f=open('xxx.pickle','wb')
#
# pickle.dump(x,f)
#
# f.close()
#
#
# f=open('xxx.pickle','rb')
# y=pickle.load(f)
# f.close()
#
# print(y)

a=np.array([[1,2,3],[4,5,6]])

b=np.max(a,axis=0)

c=np.argmax(a,axis=0)

print(b,c)