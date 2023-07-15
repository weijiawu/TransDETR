# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
# import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import moviepy
import moviepy.video.io.ImageSequenceClip
import shutil
from moviepy.editor import *

def pics2video(frames_dir="", fps=25):
    im_names = os.listdir(frames_dir)
    num_frames = len(im_names)
    frames_path = []
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(string)
        
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
#     shutil.rmtree(frames_dir)
    
def Frames2Video(frames_dir=""):
    '''  将frames_dir下面的所有视频帧合成一个视频 '''
    img_root = frames_dir      #'E:\\KSText\\videos_frames\\video_14_6'
    image = cv2.imread(os.path.join(img_root,"1.jpg"))
    h,w,_ = image.shape

    out_root = frames_dir+".avi"
    # Edit each frame's appearing time!
    fps = 20
    fourcc = VideoWriter_fourcc(*"MJPG")  # 支持jpg
    videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w, h))
    im_names = os.listdir(img_root)
    num_frames = len(im_names)
    print(len(im_names))
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( img_root, str(im_name) + '.jpg')
#         print(string)
        frame = cv2.imread(string)
        # frame = cv2.resize(frame, (w, h))
        videoWriter.write(frame)

    videoWriter.release()
    shutil.rmtree(img_root)
    
def get_annotation(video_path):
    annotation = {}
    
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})

    return annotation

def cv2AddChineseText(image, text, position, textColor=(0, 0, 0), textSize=30):
    
    
    x1,y1 = position
    x2,y2 = len(text)* textSize/2 + x1, y1 + textSize
    
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_1, [points], 1)

    image,rgb = mask_image_bg(image, mask_1,[0,0,0])
    
    
    if (isinstance(image, np.ndarray)):  # 判断是否OpenCV图片类型
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./tools/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # 转换回OpenCV格式
                    
    return image

def mask_image_bg(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.2 + mask_3d_color[mask] * 0.8
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

if __name__ == "__main__":

    
    dic_name = {
    'res_video_35':"Video_35_2_3", 'res_video_20':"Video_20_5_1", 'res_video_30':"Video_30_2_3", 'res_video_23':"Video_23_5_2", 
     'res_video_15':"Video_15_4_1",  'res_video_44':"Video_44_6_4",  'res_video_32': "Video_32_2_3", 'res_video_22': "Video_22_5_1", 
     'res_video_24':"Video_24_5_2",  'res_video_49':"Video_49_6_4", 'res_video_39' : "Video_39_2_3", 'res_video_11':"Video_11_4_1", 
     'res_video_17':"Video_17_3_1", 
     'res_video_9':"Video_9_1_1", 
     'res_video_55':"Video_55_3_2", 
     'res_video_50':"Video_50_7_4", 
     'res_video_5':"Video_5_3_2", 
     'res_video_48':"Video_48_6_4", 
     'res_video_1':"Video_1_1_2", 
     'res_video_6':"Video_6_3_2", 
     'res_video_53':"Video_53_7_4", 
    'res_video_38':"Video_38_2_3", 
    'res_video_34':"Video_34_2_3",
     'res_video_43':"Video_43_6_4"
    }
    
#     dic_name = {"Video_35_2_3": }
    
    
    
#     root = "/share/wuweijia/MyBenchMark/relabel/To30s/final_MOVText/"  e2e_TransVTS_r50_ICDAR15  e2e_TransVTS_r50_COCOTextV2  e2e_TransVTS_r50_UnrealText
    annotation_path_root ="./exps/e2e_TransVTS_r50_DSText/preds"
    txt_path = "./exps/e2e_TransVTS_r50_DSText/preds"
    
#     video_path = "/share/wuweijia/MyBenchMark/relabel/To30s/Video"
    frame_path_root = "/mmu-ocr/pub/weijiawu/Data/VideoText/MOTR/DSText/images/test"
    result_path_cls_root = "./exps/vis_DSText"
    print( os.listdir(annotation_path_root))
    
    dic_name = {}
    sequs = os.listdir(frame_path_root)
    for i in sequs:
#         dic_name.update({"res_video_"+i.split("_")[1]:i})
        dic_name.update({i:i})
#     dic_name = ["res_video_"+i.split("_")[1] for i in sequs]
    for seq in os.listdir(annotation_path_root):
        
        if "pynb" in seq:
            continue
        
        if "txt" in seq:
            continue
            
        if "batch1" not in seq:
            continue 
        print(seq) # Video_35_2_3.mp4
#         if seq!= "vitvr_yc2__y_gqHLza4Yk_0.json":
#             continue
        txt_ = os.path.join(txt_path,seq.replace(".json",".txt"))
        
        dicst_data = {}
        with open(txt_, "r") as f:
            data = f.readlines()
            for line in data:
                line = line.replace('"','').replace('\n','').split(",")
                dicst_data.update({line[0]:line[1]})
#                 
            
        result_path_cls = os.path.join(result_path_cls_root,seq.replace(".json",""))
        annotation_path = os.path.join(annotation_path_root,seq)
        frame_path = os.path.join(frame_path_root,dic_name[seq.replace(".json","")])
        
        if not os.path.exists(result_path_cls):
            os.makedirs(result_path_cls)
        print(annotation_path)
        annotation = get_annotation(annotation_path)


    #             gap = int(fps/5)
        gap=1
        lis = np.arange(0,100000,gap)
        rgbs={}
        for idx,frame_id in tqdm(enumerate(annotation.keys())):
    #             print(frame_id)

            if int(frame_id) in lis:
                frame_id_ = frame_id
                frame_id_im = frame_id
            else:
                frame_id_ = frame_id
                frame_id_im = frame_id


            while int(frame_id_) not in lis:
                frame_id_ = str(int(frame_id_)-1)
                frame_id_im = frame_id_


    #             frame_name = video_name.split(".json")[0] + "_" + frame_id_im.zfill(6) + ".jpg"
            frame_id_im = str(int(frame_id_im)-1)
            frame_name = frame_id_im.zfill(8) + ".jpg"
            frame_path_1 = os.path.join(frame_path,frame_name)

            frame = cv2.imread(frame_path_1)
            try:
                a = frame.shape[0]
            except:
                print(frame_path_1)
            annotatation_frame = annotation[frame_id_]
            ignore = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            for data in annotatation_frame:
                x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
                ID = data["ID"]
                id_content = dicst_data[str(ID)]
#                 print()
#                 id_content = "###"
                
#                 id_content = str(data["transcription"])

    #             id_content = str(data["transcription"])+ ","+ str(ID)
    #                     id_content = str(data["transcription"])
    #             is_caption = str(data["category"])


                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                mask_1 = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask_1, [points], 1)
                cv2.fillPoly(ignore, [points], 1)
                
                if ID in rgbs:
                    frame,rgb = mask_image(frame, mask_1,rgbs[ID])
                else:
                    frame,rgb = mask_image(frame, mask_1)
                    rgbs[ID] = rgb

                r,g,b = rgb[0]
                r,g,b = int(r),int(g),int(b)

                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                cv2.polylines(frame, [points], True, (r,g,b), thickness=5)
                
            for data in annotatation_frame:
                x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
                ID = data["ID"]
#                 id_content = str(data["transcription"])
#                 id_content = dicst_data[str(ID)]
                id_content = dicst_data[str(ID)]
    
                short_side = min(frame.shape[0],frame.shape[1])
                text_size = int(short_side * 0.03)
                
#                 xx1, yy1 = int(x1), int(y1) - text_size
                lists = [[int(x1), int(y1) - text_size],
                        [int(x1)-len(id_content)* text_size/2,int(y1)],
                        [int(x2),int(y2)],
                        [int(x4), int(y4)]]
                
                for i in range(4):
                    xx1, yy1 = lists[i]
                    x2,y2 = len(id_content)* text_size/2 + xx1, yy1 + text_size
                    points = np.array([[xx1, yy1], [x2, yy1], [x2, y2], [xx1, y2]], np.int32)
                    test = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    cv2.fillPoly(test, [points], 1)
                    if (test*ignore).sum()/test.sum() < 0.3:
                        break

                
                
                frame=cv2AddChineseText(frame,id_content, (int(xx1), int(yy1)),((255,255,255)), text_size)

            frame_vis_path = os.path.join(result_path_cls, frame_id+".jpg")
            cv2.imwrite(frame_vis_path, frame)
    #             video_vis_path = "./"
        pics2video(result_path_cls,fps=20)



      
        
        