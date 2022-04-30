# Metrics for multiple object text tracker benchmarking.
# https://github.com/weijiawu/MMVText-Benchmark

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import argparse
import os
import numpy as np
import copy
import motmetrics as mm
import logging
from tqdm import  tqdm
import Levenshtein
from tracking_utils.io import read_results, unzip_objs
from shapely.geometry import Polygon, MultiPoint
from motmetrics import math_util
from collections import OrderedDict
import io
string_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
def cal_similarity(string1, string2):
        if string1 == "#null" or string2 == '#null':
            return 1.0
#         string1= self.strip_points(string1)
#         string2= self.strip_points(string2)
        #在去掉标点以后再判断字符串是否为空, 防止标点字符串导致下面分母为0 
        if string1 == "" and string2 == "":
            return 1.0
        # TODO 确定是否保留，当字符串差1个字符的时候，也算对
        if Levenshtein.distance(string1, string2) == 1 :
            return 0.95
        return 1 - Levenshtein.distance(string1, string2) / max(len(string1), len(string2))
    
# /share/wuweijia/Code/VideoSpotting/TransSpotter/output/ICDAR15/test/tracks
def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="evaluation on MMVText", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str,default='/share/wuweijia/Code/VideoSpotting/TransSpotter/track_tools/Evaluation_ICDAR13/Eval_Tracking/gt'
                        , help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str,default='/share/wuweijia/Code/VideoSpotting/MOTR/exps/e2e_TransVTS_r50_ICDAR15/e2e_json',
                        help='Directory containing tracker result files')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, default="lap", help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    return parser.parse_args()


def iou_matrix_polygen(objs, gt_transcription, hyps, transcription, voc_list, max_iou=1.):
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
    for x,row in enumerate(range(m)):
        for y,col in enumerate(range(n)):
            iou = calculate_iou_polygen(objs[row], hyps[col])
            dist = iou
            
            gt_trans = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",gt_transcription[x]).upper()
            hyps_trans = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",transcription[y]).upper()
            
            min_gt_dist = 1e7
            hyps_trans_word = hyps_trans
            hyps_trans_list = []
            for voca in voc_list:
                cur_dist = Levenshtein.distance(voca, hyps_trans)
                if cur_dist < min_gt_dist:
                    min_gt_dist = cur_dist
                    hyps_trans_word = voca
                    
                    hyps_trans_list=[voca]
#                     hyps_trans_list.append(voca)
                elif cur_dist == min_gt_dist:
                    hyps_trans_list.append(voca)
            
            
            min_gt_dist = 1e7
            gt_trans_word = gt_trans
            gt_trans_list = [] 
            for voca in voc_list:
                cur_dist = Levenshtein.distance(voca, gt_trans)
                if cur_dist < min_gt_dist:
                    min_gt_dist = cur_dist
                    gt_trans_word = voca
                    gt_trans_list=[voca]
                    
                elif cur_dist == min_gt_dist:
                    gt_trans_list.append(voca)
#             if dist > max_iou and gt_trans_word!=hyps_trans_word:
#                 print(gt_trans_word,hyps_trans_word)
#                 print(gt_trans,hyps_trans)
            rec_flag = 1
            for hyps_trans_one in hyps_trans_list:
                if hyps_trans_one in gt_trans_list:
                    rec_flag = 0
#             hyps_trans = hyps_trans_word
#             gt_trans = gt_trans_word 
            
            # 更新到iou_mat
            if dist < max_iou or rec_flag:
                dist = np.nan
            
#             if dist < max_iou:
#                 dist = np.nan
                
#             if cal_similarity(gt_trans,hyps_trans)>0.5:
#                 print(gt_trans,hyps_trans)

#             if dist < max_iou or cal_similarity(gt_trans,hyps_trans)<0.5:
#                 dist = np.nan
            
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
        
        self.ignored_id = []
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)  # results_dict[fid] = [(tlwh, target_id, score)]
        
#         print(name)Video_6_3_2_GT.json Video_11_4_1_GT_voc.txt
        
        ignored_id_list={}
        
#         for frame_id in self.gt_frame_dict:
#             gt_objs = self.gt_frame_dict[frame_id]
#             for gt in gt_objs:
#                 if gt["ID"] not in ignored_id_list:
#                     ignored_id_list[gt["ID"]] = []
#                 if gt["transcription"] == "###" or "#" in gt["transcription"] or len(gt["transcription"]) < 3 or gt["transcription"]=="555":
# #                     self.ignored_id.append(gt["ID"])
#                     ignored_id_list[gt["ID"]].append("###")
#                 else:
#                     ignored_id_list[gt["ID"]].append(gt["transcription"])
        
#         for id_keys in ignored_id_list.keys():
#             list_id = ignored_id_list[id_keys]
#             num = 0
#             for i in list_id:
#                 if i == "###":
#                     num+=1
#             if num*1.0/len(list_id) >0.7:
                
#                 self.ignored_id.append(id_keys)
#         self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

        path = "/share/wuweijia/Data/ICDAR2015_video/test/Vocabulary/"
        voc_path = os.path.join(path,name.replace("_GT.json","_GT_voc.txt"))
        self.voc_list = []
        with open(voc_path, encoding='utf-8', mode='r') as f:
            for i in f.readlines():
                self.voc_list.append(i.replace("\n",""))
                
    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, trk_transcription, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        trk_transcription = np.copy(trk_transcription)
        
        gt_objs = self.gt_frame_dict[frame_id]

        gts = []
        ids = []
        transcription = []
        ignored = []
#         ignored_id = []
#         for gt in gt_objs:
# #             print(gt["transcription"])
#             if gt["transcription"] == "###" or "#" in gt["transcription"] or len(gt["transcription"]) < 3:
#                 ignored_id.append(gt["ID"])
                
        for gt in gt_objs:
            is_valid = 0
#             for i in gt["transcription"]:
#                 if i not in string_map:
#                     is_valid=1
#                     print(gt["transcription"])

            if "#" in gt["transcription"] or len(gt["transcription"]) < 3 or is_valid or gt["transcription"]=="555":
#             if gt["transcription"] == "###" or "#" in gt["transcription"]:
                ignored.append(np.array(gt["points"],dtype=np.float))
            else:
#                 print(gt["transcription"])
                gts.append(np.array(gt["points"],dtype=np.float))
                ids.append(gt["ID"])
                transcription.append(gt["transcription"])
                
        gt_objs = gts
        gt_objs = np.array(gt_objs,dtype=np.int32)
        ids = np.array(ids,dtype=np.int32)
        ignored = np.array(ignored,dtype=np.int32)
        
        
        if np.size(gt_objs) != 0:
            gt_tlwhs = gt_objs
            gt_ids = ids
            gt_transcription = transcription
        else:
            gt_tlwhs = gt_objs
            gt_ids = ids
            gt_transcription = transcription
        
        # filter 
        trk_tlwhs_ = []
        trk_ids_ = []
        trk_transcription_ = []
        
        for idx,box1 in enumerate(trk_tlwhs):
            flag = 0
            for box2 in ignored:
                iou = calculate_iou_polygen(box1, box2)
                if iou > 0.5:
                    flag=1
            if flag == 0:
                trk_tlwhs_.append(trk_tlwhs[idx])
                trk_ids_.append(trk_ids[idx])
                trk_transcription_.append(trk_transcription[idx])
                
        trk_tlwhs = trk_tlwhs_
        trk_ids = trk_ids_
        trk_transcription = trk_transcription_
        
        iou_distance = iou_matrix_polygen(gt_tlwhs,gt_transcription, trk_tlwhs, trk_transcription, self.voc_list, max_iou=0.5) # 注意 这里是越小越好！！！
        
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events
        else:
            events = None
            
        return events

    def eval_file(self, filename,transcription_path):
        self.reset_accumulator()
        #                               '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        id_to_transcription = {}
        with open(transcription_path, encoding='utf-8', mode='r') as f:
            for i in f.readlines():
                id_, transcription = i.replace('"','').split(",")
                id_to_transcription[id_] = transcription.replace("\n","")
                
        for frame_id in range(len(self.gt_frame_dict)):
            frame_id += 1
            if str(frame_id) in result_frame_dict.keys():
                trk_objs = result_frame_dict[str(frame_id)]
                
                trk_tlwhs = []
                trk_ids = []
                trk_transcription = []
                for trk in trk_objs:
#                     print(trk)
                    trk_tlwhs.append(np.array(trk["points"],dtype=np.int32))
                    trk_ids.append(np.array(trk["ID"],dtype=np.int32))
                    try:
#                         trk_transcription.append(trk["transcription"])

                        trk_transcription.append(id_to_transcription[str(trk["ID"])])
                    except:
                        trk_transcription.append("error")
                        print(trk)
            else:
                trk_tlwhs = np.array([])
                trk_ids = np.array([])
                trk_transcription = []
                
            self.eval_frame(str(frame_id), trk_tlwhs, trk_ids,trk_transcription, rtn_events=False)

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
    
    
#     logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
#     data_root = '/share/wuweijia/Data/MMVText/train/annotation'
#     result_root = '/home/guoxiaofeng/.jupyter/wuweijia/VideoTextSpotting/task123/MMVText'
    data_type = 'text'

#     seqs = ["Cls11_YS_Frames_2024021742.json","Cls11_YS_Frames_40595224195.json","Cls6_XW_Frames_40685610114.json" ,"Cls8_XJ_Frames_40159362262.json", "Cls8_XJ_Frames_41173254681.json"]
    seqs = os.listdir(args.groundtruths)
    
    filter_seqs = []
    for seq in tqdm(seqs): # tqdm(seqs):
        filter_seqs.append(seq)
#     filter_seqs = seqs[:50]
    
    accs = []
    
    for seq in tqdm(filter_seqs): # tqdm(seqs):
        # eval
        res_name = seq.replace("_GT","").replace("V","res_v")
        res_name = res_name.split("_")[0] + "_" + res_name.split("_")[1] + "_" + res_name.split("_")[2] +".json"
        result_path = os.path.join(args.tests, res_name)
        
        transcription_path = os.path.join(args.tests.replace("e2e_json","e2e_xml"), res_name.replace("json","txt"))
        
        evaluator = Evaluator(args.groundtruths, seq, data_type)
        accs.append(evaluator.eval_file(result_path,transcription_path))
        
    # metric names
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, filter_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print(strsummary)
    return summary['mota']['OVERALL']
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

if __name__ == '__main__':
    main()