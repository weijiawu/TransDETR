# -*- coding: utf-8 -*-
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from shapely.geometry import *
import json

def read_lines(p):
    """return the text in a file in lines as a list """
    p = get_absolute_path(p)
    f = open(p,'r')
    return f.readlines()

def join_path(a, *p):
    return os.path.join(a, *p)

def write_result_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = join_path(path, '%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d"%values[0]
        for v_id in range(1, len(values)):
            line += ", %d"%values[v_id]
        line += '\n'
        lines.append(line)
    write_lines(filename, lines)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def write_result_as_json(img_name,boxes_list,j_dict):
    res_name = img_name.replace("gt","res")
    # print(res_name)
    j_dict_ = []
    for b_idx, bbox in enumerate(boxes_list):

        pts = [(bbox[j],bbox[j+1]) for j in range(0,len(bbox),2)]
        # pts = [i for i in reversed(pts)]

        try:
            pdet = Polygon(pts)
        except:
            print(pts)
            assert (0), ('not a valid polygon', pts)

        pRing = LinearRing(pts)
        try:
            if not pRing.is_ccw:
                assert (0), (
                    "Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")
        except:
            print(res_name)
            print(pts)
            continue

        points = np.mat(pts)

        j_dict_.append({"points": points, "confidence": 1.0})
    j_dict.update({res_name: j_dict_})
    return j_dict

def write_lines(p, lines):
    p = get_absolute_path(p)
    make_parent_dir(p)
    with open(p, 'w') as f:
        for line in lines:
            f.write(line)

def make_parent_dir(path):
    """make the parent directories for a file."""
    parent_dir = get_dir(path)
    mkdir(parent_dir)

def exists(path):
    path = get_absolute_path(path)
    return os.path.exists(path)

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created.
    """
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
    return path

def get_dir(path):
    '''
    return the directory it belongs to.
    if path is a directory itself, itself will be return
    '''
    path = get_absolute_path(path)
    if is_dir(path):
        return path;
    return os.path.split(path)[0]

def is_dir(path):
    path = get_absolute_path(path)
    return os.path.isdir(path)


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def debug(idx, img_name, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)

    cv2.imwrite(os.path.join(output_root , img_name), res)

def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def draw_bbox(img_path, result, color=(255, 0, 0),thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger

def remove_all(s, sub):
    return replace_all(s, sub, '')


def split(s, splitter, reg = False):
    if not reg:
        return s.split(splitter)
    import re
    return re.split(splitter, s)


def replace_all(s, old, new, reg=False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(checkpoint_path, model, optimizer, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('models saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, logger, device, optimizer=None):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    logger.info('models loaded from %s' % checkpoint_path)
    return start_epoch


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back
    return newFunc


def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)


def ls(path='.', suffix=None):
    """
    list files in a directory.
    return file names in a list
    """
    path = get_absolute_path(path)
    files = os.listdir(path)

    if suffix is None:
        return files

    filtered = []
    for f in files:
        if ends_with(f, suffix, ignore_case=True):
            filtered.append(f)

    return filtered

def is_str(s):
    return type(s) == str

def to_lowercase(s):
    return str.lower(s)

def ends_with(s, suffix, ignore_case = False):
    """
    suffix: str, list, or tuple
    """
    if is_str(suffix):
        suffix = [suffix]
    suffix = list(suffix)
    if ignore_case:
        for idx, suf in enumerate(suffix):
            suffix[idx] = to_lowercase(suf)
        s = to_lowercase(s)
    suffix = tuple(suffix)
    return s.endswith(suffix)