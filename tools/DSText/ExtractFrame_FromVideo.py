#coding: utf-8
import os
import cv2
import random
from moviepy.editor import *
import numpy as np

def get_file_path_list(json_dir, postfix = [".jpg"] ):
    '''  '''
    file_path_list = []
    if os.path.exists(json_dir):
        print (json_dir)
    else:
        print ("Do not exist")
    for rt, dirs, files in os.walk(json_dir):
#         print(rt,dirs)
#         if len(dirs)>0:
#             continue
        for file in files:
            if os.path.splitext(file)[1] in postfix:
                file_path_list.append( os.path.join(rt, file ) )
    return file_path_list


def extract_frame_from_video(video_path, video_frame_save_dir ):
    '''
    :param video_path:
    :param video_frame_save_dir:
    :return: 
    '''
    Parent_dir, Video_name = os.path.split(video_path)
    VideoName_prefix, VideoName_postfix = os.path.splitext(Video_name)

    
    video_object = cv2.VideoCapture(video_path)
    fps = video_object.get(cv2.CAP_PROP_FPS)

    frame_index = 1
    while True:
        ret, frame = video_object.read()
        if ret == False:
            print("extract_frame_from_video(), extract is finished")
            return
        frame_name = "{}.jpg".format(frame_index)
        cv2.imwrite( os.path.join(video_frame_save_dir, frame_name), frame )
        frame_index += 1

def batch_extractFrame_fromVideo(VideoSet_dir="",to_video=""):
    '''
    :return: 
    '''
    if VideoSet_dir == "":
        print("Please input video folder")
        return
    VideoDir_list = get_file_path_list(VideoSet_dir, postfix = [".mp4"] )
    VideoDir_list.sort()
    print("KSText==> Extract frames from video")
    

    for Video_dir in VideoDir_list:
        print(Video_dir)
        Parent_dir, Video_name = os.path.split(Video_dir )
        Video_prefix, Video_postfix = os.path.splitext(Video_name)
        Parent_dir_new = to_video
        if not os.path.exists(Parent_dir_new):
            os.makedirs(Parent_dir_new)
        NewVideoFrame_dir = os.path.join(Parent_dir_new, Video_prefix )
        if not os.path.exists(NewVideoFrame_dir):
            os.makedirs(NewVideoFrame_dir )

        extract_frame_from_video(Video_dir, NewVideoFrame_dir)

                
        
if __name__ == "__main__":
    print ("Hello World!")
    VideoSetDir_list = []
    
    video_root = "./video/"
    output_frame_root = "./frame/"
    for i in os.listdir(video_root):
        if ".ipynb" in i:
            continue
        from_video = os.path.join(video_root,i)
        VideoSetDir_list.append(from_video)


    for VideoSet_dir in VideoSetDir_list:
        print(VideoSet_dir)
        to_video = os.path.join(output_frame_root,VideoSet_dir.split("/")[-1])
        batch_extractFrame_fromVideo(VideoSet_dir,to_video)

