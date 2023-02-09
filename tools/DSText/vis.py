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
try:
    import xml.etree.cElementTree as ET  
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def pics2video(frames_dir="", fps=25):
    im_names = os.listdir(frames_dir)
    num_frames = len(im_names)
    frames_path = []
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(string)
        
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
    shutil.rmtree(frames_dir)
    
def Frames2Video(frames_dir=""):
    img_root = frames_dir 
    image = cv2.imread(os.path.join(img_root,"1.jpg"))
    h,w,_ = image.shape

    out_root = frames_dir+".avi"
    # Edit each frame's appearing time!
    fps = 20
    fourcc = VideoWriter_fourcc(*"MJPG") 
    videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w, h))
    im_names = os.listdir(img_root)
    num_frames = len(im_names)
    print(len(im_names))
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( img_root, str(im_name) + '.jpg')

        frame = cv2.imread(string)
        videoWriter.write(frame)

    videoWriter.release()
    shutil.rmtree(img_root)
    
def getBboxesAndLabels(annotations):
    bboxes = []
    IDs = []
    words = []
    lines = []

    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        Transcription = annotation.attrib["Transcription"]
        ID = annotation.attrib["ID"]
        language = annotation.attrib["language"]
        category = annotation.attrib["category"]
        
        line = {}
        line["points"] = points
        line["ID"] = ID
        line["Transcription"] = Transcription
        lines.append(line)
        
    return lines

def get_annotation(video_path):
    annotation = {}
    utf8_parser = ET.XMLParser(encoding='utf-8')
    with open(video_path, 'r', encoding='utf-8') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)

    root = tree.getroot() 
    
    for idx,child in enumerate(root):
        lines = getBboxesAndLabels(child)
        annotation.update({idx+1:lines})


    return annotation

def cv2Text(image, text, position, textColor=(255, 255, 255), textSize=30):
    
    
    x1,y1 = position
    x2,y2 = len(text)* textSize/2 + x1, y1 + textSize
    
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_1, [points], 1)

    image,rgb = mask_image_bg(image, mask_1,[255,255,255])
    
    
    if (isinstance(image, np.ndarray)):  
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(image)

    fontStyle = ImageFont.truetype(
        "../simsun.ttc", textSize, encoding="utf-8")

    draw.text(position, text, textColor, font=fontStyle)
    
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

                    
    return image

def mask_image_bg(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    
        
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

    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    
        
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

    
    root = "./"
    frame_path_root = root +  "frame"
    annotation_path_root = root + "annotation"
    result_path_cls_root = "./vis_category"
    

    
    seqs_cls = ["Driving"]
    seqs_video = ["Video_1_5_2"]

    for cls in tqdm(os.listdir(frame_path_root)):
        frame_path = os.path.join(frame_path_root, cls)
        annotation_path = os.path.join(annotation_path_root, cls)
        result_path_cls = os.path.join(result_path_cls_root, cls)
        if not os.path.exists(result_path_cls):
            os.makedirs(result_path_cls)
        for video in tqdm(os.listdir(frame_path)):

            if ".ipynb" in video:
                continue
            annotation_path_cls_v = os.path.join(annotation_path, video.split(".mp4")[0]+"_GT.xml")
            frame_path_cls_v = os.path.join(frame_path, video.split(".mp4")[0])
            result_path_cls_video = os.path.join(result_path_cls, video.split(".mp4")[0])

            if not os.path.exists(result_path_cls_video):
                os.makedirs(result_path_cls_video)
            
            
            if os.path.isfile(result_path_cls_video+".mp4"):
                continue

    
            txt_ = os.path.join(annotation_path_cls_v.replace(".xml",".txt"))
        
            dicst_data = {}
            with open(txt_, "r") as f:
                data = f.readlines()
                for line in data:
                    line = line.replace('"','').replace('\n','').split(",")
                    dicst_data.update({line[0]:line[1]})
                
            annotation = get_annotation(annotation_path_cls_v)

            rgbs={}
            for idx,frame_id in tqdm(enumerate(annotation.keys())):
                frame_vis_path = os.path.join(result_path_cls_video, str(frame_id)+".jpg")
                if os.path.isfile(frame_vis_path):
                    continue
                    
                frame_name = str(frame_id) + ".jpg"
                frame_path_1 = os.path.join(frame_path_cls_v,frame_name)
                print(frame_path_1)
                frame = cv2.imread(frame_path_1)
                try:
                    a = frame.shape[0]
                except:
                    print(frame_path_1)
                annotatation_frame = annotation[frame_id]
                ignore = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                for data in annotatation_frame:
                    try:
                        x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
                    except:
                        print(annotation_path_cls_v)
                        assert False
                    ID = data["ID"]
                    id_content = dicst_data[str(ID)]

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
                    cv2.polylines(frame, [points], True, (r,g,b), thickness=3)
                    
                   
                for data in annotatation_frame:
                    x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
                    ID = data["ID"]

                    id_content = dicst_data[str(ID)]
        
                    short_side = min(frame.shape[0],frame.shape[1])

                    text_size = int(short_side * 0.03)

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



                    frame=cv2Text(frame,id_content, (int(xx1), int(yy1)),((0,0,0)), text_size)

                frame_vis_path = os.path.join(result_path_cls_video, str(frame_id)+".jpg")
                cv2.imwrite(frame_vis_path, frame)
            pics2video(result_path_cls_video,fps=20)





