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
import json
import scipy.io as sio

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
        object_boxes =  [int(float(i)) for i in annotation["points"]]
        ID = annotation["ID"]
        id_content = str(annotation["transcription"])
        
        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
        box, rotate = get_rotate(points)
        
        if "?" in id_content or "#" in id_content:
            continue
            
        bboxes.append(box)
        IDs.append(ID)
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

def get_annotation(video_path):
    annotation = {}
    
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})

    return annotation

def parse_xml(annotation_path,image_path):
#     utf8_parser = ET.XMLParser(encoding='gbk')
#     with open(annotation_path, 'r', encoding='gbk') as load_f:
#         tree = ET.parse(load_f, parser=utf8_parser)
#     root = tree.getroot()  # 获取树型结构的根
    
    bboxess, IDss, rotatess = [], [] , []
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    annotation = get_annotation(annotation_path)
    for idx,frame_id in tqdm(enumerate(annotation.keys())):
        annotatation_frame = annotation[frame_id]
        bboxes, IDs, rotates = \
            getBboxesAndLabels_icd13(height, width, annotatation_frame)
        bboxess.append(bboxes) 
        IDss.append(IDs)
        rotatess.append(rotates)

    return bboxess, IDss, rotatess

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def gen_data_path(path,split_train_test="train",data_path_str = "./datasets/data_path/SynthText.train"):
    
    image_path = path
    lines = []
    for cls in tqdm(os.listdir(image_path)):
        frame_path = os.path.join(image_path,cls)

        for frame_path_1 in os.listdir(frame_path):
            final_image_path = os.path.join(image_path,cls,frame_path_1)
            
            if ".txt" in frame_path_1:
                lines.append("/share/wuweijia/Data/SynthText/" + final_image_path.split("train/")[1].replace("txt","jpg") + "\n")
                print("/share/wuweijia/Data/SynthText/" + final_image_path.split("train/")[1].replace("txt","jpg") + "\n")
#         for i in range(1,len(frame_list)+1):
#             frame_real_path = path + cls + "/" + frame_path_ + "/{}.jpg".format(i) + "\n"
#             label_path = "/share/wuweijia/Data/VideoText/MOTR/BOVText/labels_with_ids/train/" + cls + "/" + frame_path_ + "/{}.txt".format(i)
#             if osp.isfile(label_path):
#                 lines.append(frame_real_path)
    write_lines(data_path_str, lines)  
    


    
from_label_root = "/mmu-ocr/weijiawu/Data/SynthText/gt.mat"
seq_root = '/mmu-ocr/weijiawu/Data/SynthText'
label_root = '/mmu-ocr/weijiawu/Data/VideoText/MOTR/SynthText/labels_with_ids/train'
mkdirs(label_root)

targets = {}
sio.loadmat(from_label_root, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

imageNames = targets['imnames']
wordBBoxes = targets['wordBB']
transcripts = targets['txt']
tid_curr = 0        
for idx in tqdm(range(len(imageNames))):
    image_path = imageNames[idx]
    word_b_boxes = wordBBoxes[idx] # 2 * 4 * num_words
    transcript = transcripts[idx]
    
    video_label_root = osp.join(label_root, image_path.split("/")[0])
    mkdirs(video_label_root)
    
    frame_path_one = osp.join(seq_root, image_path)
    img = cv2.imread(frame_path_one)
    seq_height, seq_width = img.shape[:2]
    
    
    label_fpath = osp.join(video_label_root, image_path.split("/")[1].replace("jpg","txt").replace("png","txt"))
    
    word_b_boxes = np.expand_dims(word_b_boxes, axis=2) if (word_b_boxes.ndim == 2) else word_b_boxes
    _, _, num_of_words = word_b_boxes.shape
    text_polys = word_b_boxes.transpose((2, 1, 0))
    
    if isinstance(transcript, str):
        transcript = transcript.split()
            
    words = []
    for idx,text in enumerate(transcript):
        text = text.replace('\n', ' ').replace('\r', ' ')
        words.extend([w for w in text.split(' ') if len(w) > 0])
    
    if len(words)!=len(text_polys):
        print(image_path)
        continue
        
    lines = []    
    for points,word in zip(text_polys,words):
        tid_curr += 1
        points = cv2.minAreaRect(points.reshape((int(len(points)), 2)))
        points = cv2.boxPoints(points).reshape((-1))
        box, rotate = get_rotate(points)
        x, y, w, h = box
        
        x1, y1, w1, h1 = cv2.boundingRect(points.reshape((4, 2)))
        
        x += w / 2
        y += h / 2
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f} {}\n'.format(
        tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height, rotate,x1, y1, x1 + w1, y1 + h1, word)
        lines.append(label_str)
        
    write_lines(label_fpath, lines)    
    


# gen_data_path("/share/wuweijia/Data/VideoText/MOTR/SynthText/labels_with_ids/train")

