"""
https://github.com/xingyizhou/CenterTrack
Modified by weijia wu
"""
import os
import numpy as np
import json
import cv2
import shutil
from cv2 import VideoWriter, VideoWriter_fourcc
from tqdm import tqdm
from collections import OrderedDict
from shapely.geometry import Polygon, MultiPoint
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from util.utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
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
                
            ICDAR21_DetectionTracks[frame].append({"points":text_list[:8],"ID":text_list[8],
                                                   "transcription":text_list[9]})

    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)
    
def getBboxesAndLabels_icd13(height, width, annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append(int(point.attrib["x"]))
            object_boxes.append(int(point.attrib["y"]))

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
        object_boxes = []
        for i in points:
            object_boxes.append(int(i))
        object_boxes.append(annotation.attrib["ID"])
#         points = np.array(object_boxes).reshape(-1)
#         x, y, w, h = cv2.boundingRect(points)
#         box = np.array([x, y, x+w, y, x+w, y+h, x, y+h])
#         box[0::2] = np.clip(box[0::2], 0, width - 1)
#         box[1::2] = np.clip(box[1::2], 0, height - 1)
# #         box = [str(b) for b in box]
        
#         line = str(box[0])
#         for b in range(1,len(box)):
#             line += ","
#             line += str(box[b])
        
        quality = annotation.attrib["Quality"]
        
        Transcription = annotation.attrib["Transcription"]
#         print(quality)
        if quality == "LOW":
            object_boxes.append("###")
        else:
            object_boxes.append(Transcription)
#             line += ","
#             line += Transcription
            
        
#         if "?" in Transcription or "#" in Transcription or "55" in Transcription:
#             line += ","
#             line += "###"
#         else:
#             line += ","
#             line += Transcription
           
#         line += "\n"
        
        bboxes.append(object_boxes)


    return bboxes

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


def parse_xml(annotation_path,video_path,gt_path):
    utf8_parser = ET.XMLParser(encoding='gbk')
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    annotation = {}
    for idx,child in enumerate(root):
        image_path = os.path.join(video_path, child.attrib["ID"] + ".jpg")
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
        except:
            print(image_path+"is None")
            continue
        bboxes = \
            getBboxesAndLabels_icd13(height, width, child)
        
#         gt_path_txt = os.path.join(gt_path,"{}_{}.txt".format(video_path.split("/")[-1],child.attrib["ID"]))
#         write_lines(gt_path_txt, bboxes) 
        annotation.update({str(child.attrib["ID"]):bboxes}) 
    output_cls = os.path.join(gt_path,annotation_path.split("/")[-1].replace("xml","json"))
    annotation = sort_key(annotation)
    Generate_Json_annotation(annotation,output_cls)

if __name__ == '__main__':
    
    # Use the same script for MOT16
    DATA_PATH = '/share/wuweijia/Data/ICDAR2013_video'
    OUT_PATH = "./track_tools/Evaluation_ICDAR13/Eval_Tracking/gt"
    submit = "/share/wuweijia/Code/VideoSpotting/TransTrack/output/ICDAR13/test/evaluation/"

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
        
    data_path = os.path.join(DATA_PATH, 'test/frames')
    ann_path_ = os.path.join(DATA_PATH, 'test/gt')
    seqs = os.listdir(data_path)

    for seq in tqdm(sorted(seqs)):
        ann_path = os.path.join(ann_path_, seq + "_GT.xml")
        parse_xml(ann_path,os.path.join(data_path,seq),OUT_PATH)
    
#     for i in os.listdir(submit):
#         if i not in os.listdir(OUT_PATH):
#             shutil.rmtree(os.path.join(submit,i))
        