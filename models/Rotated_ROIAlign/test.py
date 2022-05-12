'''
采用rroi_align对旋转的文字进行矫正和crop
data：2019-6-24
author:yibao2hao
注意:
    1. im_data和rois都要是cuda
    2. roi为[index, x, y, h, w, theta]
    3. 增加了batch操作支持
    4. 
'''
# from modules.rroi_align import _RRoiAlign

import torch
import cv2
import numpy as np
import math
import random
from math import sin, cos, floor, ceil
import matplotlib.pyplot as plt
from torch.autograd import Variable
from FOTS.rroi_align.functions.rroi_align import RRoiAlignFunction
from FOTS.Rotated_ROIAlign.roi_align_rotate import ROIAlignRotated
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index]

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# gt = np.asarray([[645,19],[686,13],[692,43],[651,48]])

# find_min_rect_angle(gt.reshape(-1))

if __name__=='__main__':

    path = '/home/wangjue_Cloud/wuweijia/Data/ICDAR2015/train_image/img_100.jpg'
    # path = './data/grad.jpg'
    im_data = cv2.imread(path)
    img = im_data.copy()
    im_data = torch.from_numpy(im_data).unsqueeze(0).permute(0,3,1,2)
    im_data = im_data
    im_data = im_data.to(torch.float).cuda()
    im_data = Variable(im_data, requires_grad=True)

    # plt.imshow(img)
    # plt.show()

    # 参数设置
    debug = True
    # 居民身份证的坐标位置
#     158 128 411 181 "Footpath"
#     443 128 501 169 "To"
#     64 200 363 243 "Colchester"
#     394 199 487 239 "and"
#     72 271 382 312 "Greenstead"
# ﻿645,19,686,13,692,43,651,48,Way
# 689,19,719,13,724,32,694,38,out
# 519,110,549,93,556,112,526,129,###
# 549,92,572,80,578,98,555,110,Line
# 684,172,729,160,734,176,689,188,Platform
# 730,162,739,159,742,171,733,174,###
# 744,157,749,154,753,168,748,171,###
# 759,152,767,150,772,165,764,167,###

    gt3 = np.asarray([[645,19],[686,13],[692,43],[651,48]])      # 签发机关
    gt1 = np.asarray([[684,172],[729,160],[734,176],[689,188]])     # 居民身份证
    # # gt2 = np.asarray([[205,150],[202,126],[365,93],[372,111]])     # 居民身份证
#     gt2 = np.asarray([[206,111],[199,95],[349,60],[355,80]])       # 中华人民共和国
#     gt4 = np.asarray([[312,127],[304,105],[367,88],[374,114]])       # 份证
#     gt5 = np.asarray([[133,168],[118,112],[175,100],[185,154]])      # 国徽
    # gts = [gt1, gt2, gt3, gt4, gt5]
    gts = [gt1, gt3]
    
    
    roi = []
    for i,gt in enumerate(gts):
        center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4        # 求中心点

        dw = gt[2, :] - gt[1, :]
        dh =  gt[1, :] - gt[0, :] 
        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])                    # 宽度和高度
        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])  + random.randint(-2, 2)

#         angle_gt = ( math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
#         angle_gt = -angle_gt / 3.1415926535 * 180                       # 需要加个负号
        angle_gt = find_min_rect_angle(gt.reshape(-1))
        print(angle_gt)
        roi.append([0, center[0], center[1], h, w, angle_gt])           # roi的参数

    rois = torch.tensor(roi)  
    rois = rois.to(torch.float).cuda()

    pooled_height = 44
    maxratio = rois[:,4] / rois[:,3]
    maxratio = maxratio.max().item()
    pooled_width = math.ceil(pooled_height * maxratio)
    
#     roipool = roirotate.apply(im_data.cuda(), rois.view(-1, 6), pooled_height, pooled_width, self.spatial_scale)
    roipool = RRoiAlignFunction()
#     roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)
    # 执行rroi_align操作
#     pooled_feat = roipool.apply(im_data.cuda(), rois.view(-1, 6), pooled_height, pooled_width, 1.0)
    
    pooler_rotated=ROIAlignRotated((32,192), spatial_scale = (1.), sampling_ratio = 0)
    pooled_feat=pooler_rotated(im_data.cuda(),rois.view(-1, 6))

    res = pooled_feat.pow(2).sum()
    # res = pooled_feat.sum()
    res.backward()

    if debug:
        for i in range(pooled_feat.shape[0]):
            x_d = pooled_feat.data.cpu().numpy()[i]
            x_data_draw = x_d.swapaxes(0, 2)
            x_data_draw = x_data_draw.swapaxes(0, 1)
        
            x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
#             cv2.imshow('im_data_gt %d' % i, x_data_draw)
            cv2.imwrite('./saved/rroi/res%d.jpg' % i, x_data_draw)
            
#         cv2.imwrite('./img', img)

        # 显示梯度
        im_grad = im_data.grad.data.cpu().numpy()[0]
        im_grad = im_grad.swapaxes(0, 2)
        im_grad = im_grad.swapaxes(0, 1)
        
        im_grad = np.asarray(im_grad, dtype=np.uint8)
#         cv2.imshow('grad', im_grad)
        cv2.imwrite('./saved/rroi/grad.jpg',im_grad)

        # 
        grad_img = img + im_grad
        cv2.imwrite('./saved/rroi/grad_img.jpg', grad_img)
#     cv2.waitKey(100)
#     print(pooled_feat.shape)
    
