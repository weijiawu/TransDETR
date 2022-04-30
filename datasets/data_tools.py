#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tools.py
@Time    :   2021/11/20 14:28:23
@Author  :   lizhuang05
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@Desc    :   数据预处理工具

'''
import numpy as np
import cv2
import string
import mmcv
import random
import math
# import scipy.io as scio
import json
import pyclipper
import Polygon as plg


def get_img(img_path, read_type='cv2'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img

def check(s):
    for c in s:
        if c in list(string.printable[:-6]):
            continue
        return False
    return True

def get_ann_ic15(img, gt_path):
    # h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if len(gt) < 8:
            gt = line.split('\t')
        word = gt[8].replace('\r', '').replace('\n', '')
        if len(word) == 0 or word[0] == '#':
            words.append('###')
        else:
            words.append(word)

        bbox = [int(float(gt[i])) for i in range(8)]
        # bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words

def get_ann_ic15_video(img, gt_path):
    # h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    track_id = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if len(gt) < 8:
            gt = line.split('\t')
        word = gt[8].replace('\r', '').replace('\n', '')
        if len(word) == 0 or word[0] == '#':
            words.append('###')
        else:
            words.append(word)
        
        t_id = gt[9].replace('\r', '').replace('\n', '')
        track_id.append(t_id)
        bbox = [int(float(gt[i])) for i in range(8)]
        # bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words, track_id

def get_ann_kwai_det(img, gt_path):
    # h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split('\t')
        word = gt[8].replace('\r', '').replace('\n', '')
        words.append(word)
        bbox = [int(float(gt[i])) for i in range(8)]
        # bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words

def get_ann_mtwi(img, gt_path):
    # h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = ",".join(gt[8:]).replace('\r', '').replace('\n', '')
        if len(word) == 0:
            words.append('###')
        else:
            words.append(word)
        bbox = [int(float(gt[i])) for i in range(8)]
        # bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words

def is_ver_word(bbox):
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    h = y_max - y_min
    w = x_max - x_min
    if h > w * 1.5:
        return True
    else:
        return False

def random_rot_flip(img, bboxes, words, mode):
    assert mode in [0,1,2,3,4]
    if mode == 0: 
        # 竖直flip模式
        dealed_img = cv2.flip(img, 0).copy()
        h, w = dealed_img.shape[0:2]
        # bbox y 对称
        bboxes = [[bbox[0], h-bbox[1], bbox[2], h-bbox[3], bbox[4], h-bbox[5], bbox[6], h-bbox[7]] for bbox in bboxes]
        for i, word in enumerate(words):
            if is_ver_word(bboxes[i]):
                words[i] = word[::-1]

    elif mode == 1:
        # 水平flip
        dealed_img = cv2.flip(img, 1).copy()
        h, w = dealed_img.shape[0:2]
        # bbox y 对称
        bboxes = [[w-bbox[0], bbox[1], w-bbox[2], bbox[3], w-bbox[4], bbox[5], w-bbox[6], bbox[7]] for bbox in bboxes]
        for i, word in enumerate(words):
            if not is_ver_word(bboxes[i]):
                words[i] = word[::-1]
    
    elif mode == 2:
        ori_h, ori_w = img.shape[0:2]
        # 旋转90
        dealed_img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        h, w = dealed_img.shape[0:2]
        bboxes = [[
                    w-bbox[1]/ori_h*w, bbox[0]/ori_w * h,
                    w-bbox[3]/ori_h*w, bbox[2]/ori_w * h, 
                    w-bbox[5]/ori_h*w, bbox[4]/ori_w * h, 
                    w-bbox[7]/ori_h*w, bbox[6]/ori_w * h] for bbox in bboxes]
        for i, word in enumerate(words):
            if not is_ver_word(bboxes[i]):  #如果是横排，原来是竖排，阅读顺序反序
                words[i] = word[::-1]

    elif mode == 3:
        # 旋转180
        h, w = img.shape[0:2]
        # 旋转90
        dealed_img = cv2.rotate(img, cv2.cv2.ROTATE_180) 
        bboxes = [[w-bbox[0], h-bbox[1], w-bbox[2], h-bbox[3], w-bbox[4], h-bbox[5], w-bbox[6], h-bbox[7]] for bbox in bboxes]
        for i, word in enumerate(words):
            # 无论横竖都要反序
            words[i] = word[::-1]

    elif mode == 4:
        # 旋转270
        ori_h, ori_w = img.shape[0:2]
        dealed_img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 
        h, w = dealed_img.shape[0:2]
        bboxes = [[
                    bbox[1]/ori_h*w, h - bbox[0]/ori_w * h,
                    bbox[3]/ori_h*w, h - bbox[2]/ori_w * h, 
                    bbox[5]/ori_h*w, h - bbox[4]/ori_w * h, 
                    bbox[7]/ori_h*w, h - bbox[6]/ori_w * h] for bbox in bboxes]
        for i, word in enumerate(words):
            if is_ver_word(bboxes[i]):  #如果是横排，原来是竖排，阅读顺序反序
                words[i] = word[::-1]
    else:
        raise "Not a valida mode: {}!".format(mode)
    return dealed_img,  bboxes, words

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs, max_angle=10):
    
    angle = random.random() * 2 * max_angle - max_angle

    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs

def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def random_scale(img, min_size, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)
    # print (h_scale, w_scale, h_scale / w_scale)

    img = scale_aligned(img, h_scale, w_scale)
    return img

def update_word_mask(labels, instance, instance_before_crop, word_mask, mask_iou=0.9):

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        # 这里设置了只要切割文字超过10% 就mask掉
        if float(np.sum(ind)) / np.sum(ind_before_crop) > mask_iou:
            continue  # 
        word_mask[label] = 0   

    return word_mask

def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)
    return shrinked_bboxes

def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK', use_ctc=False):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    elif voc_type == 'CHINESE':
        with open('/share/lizhuang05/code/pan_pp.pytorch_dev/data/keys.json', encoding='utf-8') as j:
            voc = json.load(j)
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    # voc.append(EOS). # CTC 不存在最后一个位置
    if use_ctc:
        voc.append(UNKNOWN)
        voc.append(PADDING)
    else:
        # attn
        voc.append(EOS)
        voc.append(PADDING)
        voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def resize_fix(imgs, shape):
    for i in range(1, len(imgs)):
        img = imgs[i]
        # 注意 这里不能使用双线行差值
        img_fixed = cv2.resize(img, shape, interpolation=cv2.INTER_NEAREST)
        imgs[i] = img_fixed
    # 图像预处理需要使用双线性
    imgs[0] = cv2.resize(imgs[0], shape)
    return imgs

def random_crop(imgs, max_ratio=0.4):

    h, w = imgs[0].shape[0:2]
    assert 0 <= max_ratio < 1.0
    # 随机选取一个裁剪比例
    crop_ratio = random.uniform(0, max_ratio)
    # 随机选取裁剪w/h/both
    mode = random.randint(1, 3)
    if mode == 1:
        # h
        spare_h = int((1-crop_ratio) * h)
        spare_w = w
        st_h = random.randint(0, h-spare_h) if h-spare_h > 0 else 0
        st_w = 0
    elif mode == 2:
        # w
        spare_h = h
        spare_w = int((1-crop_ratio) * w)
        st_h = 0
        st_w = random.randint(0, w-spare_w) if w-spare_w > 0 else 0
    else:
        # both
        spare_h = int((1-crop_ratio) * h)
        spare_w = int((1-crop_ratio) * w)
        st_h = random.randint(0, h-spare_h) if h-spare_h > 0 else 0
        st_w = random.randint(0, w-spare_w) if w-spare_w > 0 else 0
    # 开始裁剪
    n_imgs = []
    for idx in range(len(imgs)):
        img = imgs[idx][st_h: st_h+spare_h, st_w: st_w+spare_w]
        n_imgs.append(img)
    return n_imgs





def random_crop_padding_4typing(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs
    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    i = random.randint(0, h - t_h) if h - t_h > 0 else 0
    j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs

def random_crop_padding(imgs, target_size, crop_word_ratio=0.375):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > crop_word_ratio and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0 # 如果>0说明 文字区域在外面
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0    # 起始x
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0    # 起始y
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs