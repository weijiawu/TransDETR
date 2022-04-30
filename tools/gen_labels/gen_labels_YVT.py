import os.path as osp
import os
import numpy as np
from util.utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math
from tqdm import tqdm

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return np.array(new_box)


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
            
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    vertices = adjust_box_sort(vertices)
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
    return angle_list[best_index] / 180 * math.pi

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
#     print(v)
#     print(anchor)
    if anchor is None:
#         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()],[v[1].sum()]])/4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

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

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min,x_max-x_min , y_max-y_min]),theta
    
def getBboxesAndLabels_icd13(height, width, annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    rotates = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(float(point.attrib["x"])), int(float(point.attrib["y"]))])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
        box, rotate = get_rotate(points)
        
#         x, y, w, h = cv2.boundingRect(points.reshape((4, 2)))
#         box = np.array([x, y, w, h])
#         label = annotation.attrib["Transcription"]
#         Transcription = annotation.attrib["Transcription"]
        
        
        quality = annotation.attrib["Quality"]
        Transcription = annotation.attrib["Transcription"]
#         if quality == "LOW":
#             continue   
        if "?" in Transcription or "#" in Transcription:
            continue
            
            
        bboxes.append(box)
        IDs.append(annotation.attrib["ID"])
        rotates.append(rotate)

    if bboxes:
        bboxes = np.array(bboxes, dtype=np.float32)
        # filter the coordinates that overlap the image boundaries.
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
        IDs = np.array(IDs, dtype=np.int64)
        rotates = np.array(rotates, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)
        # polygon_point = np.zeros((0, 8), dtype=np.int)
        IDs = np.array([], dtype=np.int64)
        rotates = np.array([], dtype=np.float32)

    return bboxes, IDs, rotates

def parse_xml(annotation_path,image_path):
    utf8_parser = ET.XMLParser(encoding='gbk')
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    bboxess, IDss, rotatess = [], [] , []
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

            
    for idx,child in enumerate(root):
        bboxes, IDs, rotates = \
            getBboxesAndLabels_icd13(height, width, child)
        bboxess.append(bboxes) 
        IDss.append(IDs)
        rotatess.append(rotates)
    return bboxess, IDss, rotatess

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def gen_data_path(path,split_train_test="train",data_path_str = "./datasets/data_path/YVT.train"):
    
    image_path = os.path.join(path,"images",split_train_test)
    lines = []
    for video_name in os.listdir(image_path):
        frame_path = os.path.join(image_path,video_name)
        frame_list = []
        for frame_path_ in os.listdir(frame_path):
            if ".jpg" in frame_path_:
                frame_list.append(frame_path_)
        for i in range(0,len(frame_list)):
            frame_real_path = "YVT/images/train/" + video_name + "/{}f{}.jpg".format(video_name,str(i).zfill(4)) + "\n"
            lines.append(frame_real_path)
    write_lines(data_path_str, lines)  
    

from_label_root = '/share/wuweijia/Data/VideoText/YVT/YVT_train'
seq_root = '/share/wuweijia/Data/VideoText/MOTR/YVT/images/train'
label_root = '/share/wuweijia/Data/VideoText/MOTR/YVT/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]
# seqs = ['ADL-Rundle-6', 'ETH-Bahnhof', 'KITTI-13', 'PETS09-S2L1', 'TUD-Stadtmitte', 'ADL-Rundle-8', 'KITTI-17',
#         'ETH-Pedcross2', 'ETH-Sunnyday', 'TUD-Campus', 'Venice-2']

gen_data_path(path="/share/wuweijia/Data/VideoText/MOTR/YVT")

tid_curr = 0
tid_last = -1
for seq in tqdm(seqs):
    image_path_frame = osp.join(seq_root,seq)
    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)
    
    ann_path = os.path.join(from_label_root, seq + "_GT.xml")
#     print(seq)
    if "ipynb" in seq:
        continue
    bboxess, IDss, rotatess = parse_xml(ann_path,osp.join(image_path_frame,os.listdir(image_path_frame)[0]))
    
    ID_list = {}
    
    for i in range(len(IDss)):
        frame_id = i
        label_fpath = osp.join(seq_label_root, '{}f{}.txt'.format(seq,str(frame_id).zfill(4)))
        frame_path_one = osp.join(image_path_frame,"{}f{}.jpg".format(seq,str(frame_id).zfill(4)))
        img = cv2.imread(frame_path_one)
        seq_height, seq_width = img.shape[:2]
        
        lines = []
        if IDss[i] == []:
            with open(label_fpath, 'w') as f:
                pass
                continue
#                 f.write(label_str)
        for bboxes,IDs,rotates in zip(bboxess[i],IDss[i],rotatess[i]):
            track_id = int(IDs)            
            x, y, w, h = bboxes
            
            if track_id not in ID_list:
                tid_curr += 1
                ID_list[track_id] = tid_curr
                real_id = tid_curr
            else:
                real_id = ID_list[track_id]
            x += w / 2
            y += h / 2
#             label_fpath = osp.join(seq_label_root, '{}.txt'.format(frame_id))
            
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            real_id, x / seq_width, y / seq_height, w / seq_width, h / seq_height,rotates)
            lines.append(label_str)
            
        write_lines(label_fpath, lines)     
