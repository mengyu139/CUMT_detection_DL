from Net.ssd_vgg.data.config import v2
import torch
from math import sqrt as sqrt
from itertools import product as product
import cv2
import numpy as np

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = 300
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = 6
        self.variance = [0.1, 0.2]
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = True
        self.version = 'v2'
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                step_x = step_y = self.image_size/k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w+0.5) * step_x)
                    c_y = ((h+0.5) * step_y)
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size  # 300
                    # aspect_ratio: 1,
                    # size: min_size
                    mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                             (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    if self.max_sizes[i] > 0:
                        # aspect_ratio: 1
                        # size: sqrt(min_size * max_size)/2
                        c_w = c_h = sqrt(self.min_sizes[i] *
                                         self.max_sizes[i])/2
                        mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                 (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    # rest of prior boxes
                    for ar in self.aspect_ratios[i]:
                        if not (abs(ar-1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar)/2
                            c_h = self.min_sizes[i] / sqrt(ar)/2
                            mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                     (c_x+c_w)/s_k, (c_y+c_h)/s_k]

        # [8732,4]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output




if __name__=="__main__":
    m = PriorBox(v2)
    out = m.forward()

    print(out.shape)
#
# if __name__=="__main__":
#
#
#     img = cv2.imread('1.jpg')
#
#     W=img.shape[1]
#     H=img.shape[0]
#
#     for i in [8200,8001]:
#
#         pt1_x = int(( out[i,0]-0.5*out[i,2])*W)
#         pt1_y = int(( out[i,1]-0.5*out[i,3])*H)
#         pt2_x = int(( out[i,0]+0.5*out[i,2])*W)
#         pt2_y = int(( out[i,1]+0.5*out[i,3])*H)
#
#         print(pt1_x,pt1_y,pt2_x,pt2_y)
#
#         x=int(out[i,0]*W)
#         y=int(out[i,1]*H)
#         cv2.circle(img,center=(x,y),radius=3,color=(0,255,0),thickness=-1)
#
#         cv2.rectangle(img,pt1=(pt1_x,pt1_y),pt2=(pt2_x,pt2_y),color=(0,0,255),thickness=1)
#
#
#
#     cv2.imshow('img',img)
#
#     cv2.waitKey(0)
#
#



#
#
#
# print(m.forward().size())