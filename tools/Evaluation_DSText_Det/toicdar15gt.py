# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
from cv2 import VideoWriter, VideoWriter_fourcc
from tqdm import tqdm
from collections import OrderedDict
from shapely.geometry import Polygon, MultiPoint
from functools import reduce
import operator
import math
import json
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from tqdm import tqdm


class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name,'wb')
        pickle.dump(data_dict,output)
        output.close()

    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict

    #Python语言特定的序列化模块是pickle，但如果要把序列化搞得更通用、更符合Web标准，就可以使用json模块
    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io
        with io.open(file_name, 'w', encoding='utf-8') as fp:
            # fp.write(unicode(json.dumps(data_dict, ensure_ascii=False, indent=4) ) )  #可以解决在文件里显示中文的问题，不加的话是 '\uxxxx\uxxxx'
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4) ) )


    @staticmethod
    def file2dict_json(file_name):
        import json, io
        with io.open(file_name, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
        return data_dict
    
def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir):
    '''   '''
    ICDAR21_DetectionTracks = {}
    text_id = 1
    for frame in TL_Cluster_Video_dict.keys():
        ICDAR21_DetectionTracks[frame] = []
        
        for text_list in TL_Cluster_Video_dict[frame]:
                
            ICDAR21_DetectionTracks[frame].append(text_list)

    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)

def get_bboxes( data_polygt):
    bboxes = []
    text_tags = []
    dic = {}
    
    with open(data_polygt, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split('\t')
            try:
                point = list(map(float, params[1:]))
                key = params[0].split("\\")[0]
                if key in dic.keys():
                    dic[key].append(point)
                else:
                    dic[key] = [point]

            except:
                print('load label failed on {}'.format(line))
    return dic

# 普通 dict 插入元素时是无序的，使用 OrderedDict 按元素插入顺序排序
# 对字典按key排序, 默认升序, 返回 OrderedDict
def sort_key(old_dict, reverse=False):
    """对字典按key排序, 默认升序, 不修改原先字典"""
    # 先获得排序后的key列表
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 key 列表
    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict

def order_points(pts):
    ''' sort rectangle points by clockwise '''
    sort_x = pts[np.argsort(pts[:, 0]), :]
    
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:,1])[::-1], :]
    # Right sort
    Right = Right[np.argsort(Right[:,1]), :]
    
    return np.concatenate((Left, Right), axis=0)

def order_points_1(coords):
#     coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

def calculate_iou_polygen(bbox1, bbox2):
    '''
    :param bbox1: [x1, y1, x2, y2, x3, y3, x4, y4]
    :param bbox2:[x1, y1, x2, y2, x3, y3, x4, y4]
    :return:
    '''
    bbox1 = np.array([bbox1[0], bbox1[1],bbox1[2], bbox1[3],bbox1[4], bbox1[5],bbox1[6], bbox1[7]]).reshape(4, 2)
    poly1 = Polygon(bbox1).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    bbox2 = np.array([bbox2[0], bbox2[1],bbox2[2], bbox2[3],bbox2[4], bbox2[5],bbox2[6], bbox2[7]]).reshape(4, 2)
    poly2 = Polygon(bbox2).convex_hull
    if poly1.area  < 0.01 or poly2.area < 0.01:
        return 0.0
    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area
        iou = float(inter_area) / union_area
    return iou

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    
    if summatory > 0:
        return 0 
    return 1

def getBboxesAndLabels_icd13(annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    rotates = []
    bboxes_box = []
    words = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
#         points_rotate = cv2.minAreaRect(points.reshape((4, 2)))
#         # 获取矩形四个顶点，浮点型
#         points_rotate = cv2.boxPoints(points_rotate).reshape((-1))
#         rotate_box, rotate = get_rotate(points_rotate)
        
#         x, y, w, h = cv2.boundingRect(points.reshape((4, 2)))
#         box = np.array([x, y, w, h])
        

#         quality = annotation.attrib["Quality"]
        Transcription = annotation.attrib["Transcription"]
        if Transcription == "##DONT#CARE##":
            Transcription = "###"   

        words.append(Transcription)    
        bboxes_box.append(points)

    if bboxes_box:
        bboxes_box = np.array(bboxes_box, dtype=np.float32)
    else:
        bboxes_box = np.zeros((0, 8), dtype=np.float32)
        words = []

    return bboxes_box, words

def parse_xml(annotation_path):
    
    try:
        utf8_parser = ET.XMLParser(encoding='gbk')
        with open(annotation_path, 'r', encoding='gbk') as load_f:
            tree = ET.parse(load_f, parser=utf8_parser)
        root = tree.getroot()  # 获取树型结构的根
    except:
        utf8_parser = ET.XMLParser(encoding='utf-8')
        with open(annotation_path, 'r', encoding='utf-8') as load_f:
            tree = ET.parse(load_f, parser=utf8_parser)
        root = tree.getroot()  # 获取树型结构的根
    bboxess, IDss, rotatess, wordss,orignial_bboxess = [], [] , [], [], []

#     img = cv2.imread(image_path)
#     height, width = img.shape[:2]

            
    for idx,child in enumerate(root):
        bboxes_box, words = \
            getBboxesAndLabels_icd13(child)
        bboxess.append(bboxes_box) 
        wordss.append(words)

        
    return bboxess, wordss

def read_text_results(filename):
    """
    读取json文件
    """
    results_dict = {}
    with open(filename, 'r', encoding='utf-8') as j:
        results_dict = json.load(j)
    return results_dict

if __name__ == "__main__":

            
    
#     gt = "/new_share/wuweijia/MyBenchMark/MMVText/Metrics/Detection_Metric/icdar15_evaluate/MMVText_gt"
#     res = "/new_share/wuweijia/MyBenchMark/MMVText/Metrics/Detection_Metric/icdar15_evaluate/MMVText_res"
    gt = "./ch3_test_gt_fix"
    res = "./preds"
    
    gt_out = "./gt_icdar/"
    res_out = "./pred_icdar/"
    
    filter_seqs = os.listdir(res)
    
    idxx = 1
    for seq in tqdm(filter_seqs):
        result_path = os.path.join(res, seq)
        gt_path = os.path.join(gt, seq.replace("res_","").replace(".xml","_GT.xml"))
        
        bboxess, wordss = parse_xml(result_path)
        bboxess_gt, wordss_gt = parse_xml(gt_path)
        
       
        
        for i in range(len(wordss)):
            frame_id = i + 1
            
            res_file = os.path.join(res_out,"res_"+seq.replace("res_","").replace(".xml","_")+str(frame_id)+".txt")
            with open(res_file, 'w') as f:
                for bboxes,word in zip(bboxess[i],wordss[i]):
                    points = np.array([int(float(ccc)) for ccc in bboxes])
#                     print(points)
                    points = np.reshape(points, (4,2))
                    points = order_points(points)
                    points = np.reshape(points, -1)
                    if not validate_clockwise_points(points):
                        print("invalid")
                        continue
                    strResult = ','.join(
                        [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                         str(points[6]), str(points[7])])  + '\r\n'
                    f.write(strResult)
                    
#         for i in range(len(bboxess_gt)):
#             frame_id = i + 1        
            gt_file = os.path.join(gt_out,"gt_"+seq.replace("res_","").replace(".xml","_")+str(frame_id)+".txt")
            with open(gt_file, 'w') as f:
                for bboxes,word in zip(bboxess_gt[i],wordss_gt[i]):
                    points = np.array([int(float(ccc)) for ccc in bboxes])
                    points = np.reshape(points, (4,2))
                    points = order_points(points)
                    points = np.reshape(points, -1)
                    if not validate_clockwise_points(points):
                        print("invalid")
                        continue
                    strResult = ','.join(
                        [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                         str(points[6]), str(points[7])]) + ","+ word + '\r\n'
                    f.write(strResult)
#                 idxx+=1
    
    
    
    