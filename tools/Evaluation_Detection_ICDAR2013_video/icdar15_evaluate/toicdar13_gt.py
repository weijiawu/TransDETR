# Metrics for multiple object text tracker benchmarking.
# https://github.com/weijiawu/MMVText-Benchmark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import copy
import motmetrics as mm
import logging
from tqdm import  tqdm
from shapely.geometry import Polygon, MultiPoint
from collections import OrderedDict
import io
import json

# /share/wuweijia/Code/VideoSpotting/MOTR/exps/e2e_TransVTS_r50_ICDAR15/jons
def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="evaluation on MMVText", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str,default='/share/wuweijia/Code/VideoSpotting/TransSpotter/track_tools/Evaluation_ICDAR13/Eval_Tracking/gt'
                        , help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str,default='/share/wuweijia/Code/VideoSpotting/MOTR/exps/e2e_TransVTS_r50_ICDAR15/jons',
                        help='Directory containing tracker result files')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, default="lap", help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    return parser.parse_args()


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = list(zip(*objs))[:3]
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 8)
    return tlwhs, ids, scores

def iou_matrix_polygen(objs, hyps, max_iou=0.5):
    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asfarray(objs)  # m
    hyps = np.asfarray(hyps)  # n
    m = objs.shape[0]
    n = hyps.shape[0]
    # 初始化一个m*n的矩阵
    dist_mat = np.zeros((m, n))
    
#     print(objs)
    assert objs.shape[1] == 8
    assert hyps.shape[1] == 8
    # 开始计算
    for row in range(m):
        for col in range(n):
            iou = calculate_iou_polygen(objs[row], hyps[col])
            dist = iou
            # 更新到iou_mat
            if dist < max_iou:
                dist = np.nan
            
            dist_mat[row][col] = dist
    return dist_mat

def calculate_iou_polygen(bbox1, bbox2):
    '''
    :param bbox1: [x1, y1, x2, y2, x3, y3, x4, y4]
    :param bbox2:[x1, y1, x2, y2, x3, y3, x4, y4]
    :return:
    '''
    bbox1 = np.array([bbox1[0], bbox1[1],
                      bbox1[6], bbox1[7],
                      bbox1[4], bbox1[5],
                      bbox1[2], bbox1[3]]).reshape(4, 2)
    poly1 = Polygon(bbox1).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    bbox2 = np.array([bbox2[0], bbox2[1],
                      bbox2[6], bbox2[7],
                      bbox2[4], bbox2[5],
                      bbox2[2], bbox2[3]]).reshape(4, 2)
    poly2 = Polygon(bbox2).convex_hull
    if poly1.area  < 0.01 or poly2.area < 0.01:
        return 0.0
    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou = float(inter_area) / union_area
    return iou


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    elif data_type in ('text'):
        read_fun = read_text_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))
    return read_fun(filename, is_gt, is_ignore)

def read_text_results(filename, is_gt, is_ignore):
    """
    读取json文件
    """
    results_dict = {}
    if is_ignore:
        return results_dict
    with open(filename, 'r', encoding='utf-8') as j:
        results_dict = json.load(j)
    return results_dict

"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():  # 1,486,493,433,209,165,1,7
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())
                # 矩形面积
                box_size = float(linelist[4]) * float(linelist[5])

                if is_gt:  # 走的这里 但是名字里面没有MOT
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                #if box_size > 7000:
                #if box_size <= 7000 or box_size >= 15000:
                #if box_size < 15000:
                    #continue

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict

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

class Evaluator(object):
    #           SVTS/images/val   video_5_5  mot
    # data_root: label path
    # seq_name: video name
    # data type: "text"
    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type in ('mot', 'text')
        if self.data_type == 'mot':
            gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
            
        else:
            name = self.seq_name.replace("Frames","GtTxtsR2Frames")
#             name = name.split("_")[0]+"_"+name.split("_")[1] + "_" +name.split("_")[2] + "/" + name.split("_")[3]
            gt_filename = os.path.join(self.data_root,name) 
#             gt_filename = self.seq_name
            
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)  # results_dict[fid] = [(tlwh, target_id, score)]
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)


    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, D_seq, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        gt_objs = self.gt_frame_dict[frame_id]
        
        res_path = "/share/wuweijia/Code/VideoSpotting/MOTR/tools/Evaluation_Detection_ICDAR2013_video/icdar15_evaluate/res" + "/{}{}.txt".format(D_seq,frame_id)
        gt_path = "/share/wuweijia/Code/VideoSpotting/MOTR/tools/Evaluation_Detection_ICDAR2013_video/icdar15_evaluate/gt" + "/{}{}.txt".format(D_seq,frame_id)
        
        gts = []
        with open(gt_path, 'w') as f:
            if gt_objs is not None:
                
                for gt in gt_objs:
                    poly = np.array(gt["points"]).astype(np.int32)
                    points = np.reshape(poly, -1)
                    if not validate_clockwise_points(points):
                        print("invalid")
                        continue
                        
                    x_min = points[::2].min()
                    x_max = points[::2].max()
                    y_min = points[1::2].min()
                    y_max = points[1::2].max()
                    points = [x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max]
                    
                    if gt["transcription"] == "###":
                        content = "###"
                    else:
                        content = "aaa"
                    strResult = ','.join(
                                    [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                                     str(points[6]), str(points[7])]) + ","+ content + '\r\n'
                    f.write(strResult)
        
        
        with open(res_path, 'w') as f:
            if trk_tlwhs is not None:
                for box1 in trk_tlwhs:
                    poly = np.array(box1).astype(np.int32)
                    points = np.reshape(poly, -1)
                    if not validate_clockwise_points(points):
                        print("invalid")
                        continue
                        
                    x_min = points[::2].min()
                    x_max = points[::2].max()
                    y_min = points[1::2].min()
                    y_max = points[1::2].max()
                    points = [x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max]
                    
                    strResult = ','.join(
                                    [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                                     str(points[6]), str(points[7])]) + '\r\n'
                    f.write(strResult)

    

    def eval_file(self, filename,sqe):
        self.reset_accumulator()
        #                               '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        
        for frame_id in range(len(self.gt_frame_dict)):
            frame_id += 1
            if str(frame_id) in result_frame_dict.keys():
                trk_objs = result_frame_dict[str(frame_id)]

                trk_tlwhs = []
                trk_ids = []
                for trk in trk_objs:
                    trk_tlwhs.append(np.array(trk["points"],dtype=np.int32))
                    trk_ids.append(np.array(trk["ID"],dtype=np.int32))
            else:
                trk_tlwhs = np.array([])
                trk_ids = np.array([])
            
            self.eval_frame(str(frame_id), trk_tlwhs, trk_ids,sqe, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota','motp', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )
        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
        
def main():
    args = parse_args()
    
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver
        mm.lap.default_solver = 'lap'

    data_type = 'text'

    seqs = os.listdir(args.groundtruths)
    
    filter_seqs = []
    for seq in tqdm(seqs): # tqdm(seqs):
        filter_seqs.append(seq)
    
    accs = []
    
    for seq in tqdm(filter_seqs): # tqdm(seqs):
        # eval  (D_seq.split("_")[0]+"_"+D_seq.split("_")[1]+".json").replace("Video","res_video")
        D_seq = seq.replace("_GT","")
        result_path = os.path.join(args.tests, D_seq)
        evaluator = Evaluator(args.groundtruths,seq , data_type)
        evaluator.eval_file(result_path,D_seq)
        
    

if __name__ == '__main__':
    main()