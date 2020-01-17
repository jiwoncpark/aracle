# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:21:55 2020

@author: alexf
"""
import numpy as np
import os
import skvideo.io
from datetime import date, time, datetime, timedelta
from PIL import Image, ImageOps, ImageDraw, ImageFont

X_DATA_SERIES = 'hmi.M.720s.'
TIME_INTERVAL = timedelta(minutes = 60)
FRAME_RATE = 30
TXT_BOX_HEIGHT = 15


def verify_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory {} does not exist. Creating...".format(dir_name))

def verify_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            raise ValueError('One or more of the inputed directories does not exist. The directory that threw this error is {}'.format(directory))

def generate_daterange(start,end):
    list_length = int(((end - start) / TIME_INTERVAL) + 1)
    return [start + (TIME_INTERVAL * i) for i in range(list_length)]

def generate_dates(date_intervals_list):
    date_list = []
    for start,end in date_intervals_list:
        date_list.append(generate_daterange(start,end))
    date_list = [item for sublist in date_list for item in sublist]
    return date_list

def norm_im(im):
    mn = np.amin(im)
    mx = np.amax(im)
    norm_im = (im-mn)/(mx-mn)
    return norm_im

def npy_of_text(text,width,height):
    img = Image.new('RGB', (height, width), color=0)   #black background
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", TXT_BOX_HEIGHT)
    color = 'rgb(255, 255, 255)' # white text
    draw.text((0,0), text, fill=color, font=font)
    im = np.array(img)
    return norm_im(im)

def add_label_to_frame(frame, text):
    txt_box_width = frame.shape[1]
    frame[0:TXT_BOX_HEIGHT,0:txt_box_width] = npy_of_text(text,TXT_BOX_HEIGHT,txt_box_width)
    return frame

def generate_frame(Xfile_dir,Yfile_dir,timestamp,size):
    X_im = np.load(Xfile_dir)
    Y_im = np.max(np.load(Yfile_dir),axis = 0)
    X_im_3d = np.dstack([X_im,X_im,X_im])
    Y_im_3d = np.dstack([Y_im,Y_im,Y_im])
    frame = size
    composite_frame = np.zeros((size,size*2,3))
    composite_frame[0:frame,0:frame] = add_label_to_frame(X_im_3d,timestamp)
    composite_frame[0:frame,frame:frame*2] = add_label_to_frame(Y_im_3d,timestamp)
    return composite_frame

def generate_video(Xdir,Ydir,Vid_dir,datetime_list,vid_name,size):
    verify_dirs([Xdir,Ydir,Vid_dir])
    image_list = []
    for current_time in datetime_list:
        timestamp = current_time.strftime('%Y%m%d_%H%M%S') + '_' + str(size) + '.npy' 
        Xfilename = timestamp
        Yfilename = timestamp
        Xfile_dir = os.path.join(Xdir,Xfilename)
        Yfile_dir = os.path.join(Ydir,Yfilename)
        if (os.path.exists(Xfile_dir) and os.path.exists(Yfile_dir)):
            frame = generate_frame(Xfile_dir,Yfile_dir,timestamp,size)
            image_list.append(frame)
    if image_list:
        output_dir = os.path.join(Vid_dir,vid_name)
        image_list = np.stack(image_list)
        skvideo.io.vwrite(output_dir,image_list)

savedir = 'C:/Users/alexf/Desktop/HMI_Data/'
verify_dir(savedir)
Xdata_dir = os.path.join(savedir, 'Xdata', 'uncropped_hmis_256')
Ydata_dir = os.path.join(savedir, 'Ydata', 'filled_ellipse_uncropped_256')
start = datetime(2010,5,1,0,0,0)
end =  datetime(2010,6,1,0,0,0)
dates = generate_daterange(start,end)
vid_name = 'UNCROPPED_ELLIPSES' + start.strftime('%Y%m%d_%H%M%S') + '-' + end.strftime('%Y%m%d_%H%M%S') + '.mp4'
generate_video(Xdata_dir,Ydata_dir,savedir,dates,vid_name,256)