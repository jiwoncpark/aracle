import numpy as np
import astropy.units as u
import os, fnmatch
import sunpy.coordinates
import pandas as pd
import shutil
import math
import random
import pickle
import warnings
import sunpy.map
import sys
import imageio
import matplotlib as plt
import matplotlib.pyplot as pyplot
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from PIL import Image, ImageOps, ImageDraw, ImageFont
from astropy.io import fits
from skimage import img_as_ubyte

TIME_INTERVAL = timedelta(minutes = 60)
CROP_FACTOR = .1 #CROP_FACTOR * 2 is the portion you crop
CROP_INDEX = 4096 * CROP_FACTOR
SD_FACTOR = 3#SD_FACTOR is the number of standard deviations within the radius of the circle
THRESHOLD = np.exp(-0.5 * SD_FACTOR**2)#there will be no values less than THRESHOLD in gaussian disks

def load_df(df_path):#load data text file
    df = pickle.load(open(df_path,'rb'))
    #df = check_df(df)
    times = df['T_REC'].sort_values(ascending=True)
    print('The dataframe you have loaded contains HARPS from {} to {}'.format(times.iloc[0],times.iloc[-1]))
    return df
   
def generate_max_radius_rows(df):
    max_radius_rows = df[df.groupby(['HARPNUM'])['NPIX'].transform(max) == df['NPIX']]#fix iloc issue
    max_radius_rows = max_radius_rows.reset_index(drop = True)
    return max_radius_rows#dataframe containing each harp at its max size

def generate_radius_dict(max_radius_rows):
    max_radius_rows['RADIUS'] = max_radius_rows.loc[:,['NAXIS1', 'NAXIS2']].max(axis=1)/2#find the max height/width then divide by 2
    radius_dict = pd.Series(max_radius_rows.RADIUS.values,index=max_radius_rows.HARPNUM).to_dict()#dict of each max radius
    return radius_dict

def verify_dir(dir_name):
    if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Directory {} does not exist. Creating...".format(dir_name))

def generate_sizedirs(sizes,dir_name):
    for size in sizes:
        data_dir = dir_name + '_' + str(size)
        verify_dir(data_dir)

def get_coords(row,obstimestring):
    hpc1 = SkyCoord(float(row['LON_FWT'])*u.deg, float(row['LAT_FWT'])*u.deg, frame=frames.HeliographicStonyhurst, obstime=obstimestring)
    hpc_out = sunpy.coordinates.Helioprojective(observer='earth', obstime=obstimestring) # arcsec
    hpc2 = hpc1.transform_to(hpc_out) # convert to arcsecond
    xc = (hpc2.Tx / u.arcsec) # stripping units
    yc = (hpc2.Ty / u.arcsec)
    xc = (xc/float(row['CDELT1'])) + float(row['IMCRPIX1']) # convert to pixel value
    yc = (yc/float(row['CDELT2'])) + float(row['IMCRPIX2'])
    return xc,yc

def get_radius(row,radius_type):
    if radius_type == 'real-time':
        NPIX = row['NPIX']
        radius = np.sqrt(NPIX / math.pi)
        return radius
    if radius_type == 'bounding-box':
        radius = .5 * (row['NAXIS1'] + row['NAXIS2'])
        return radius
    if radius_type == 'max-size':
        radius = radius_dict[row['HARPNUM']]
        return radius
    raise ValueError('You have selected an invalid dimension type: {}. Please select: real-time,bounding-box,or max-size'.format(radius_type))


def get_width_height(row,width_height_type):
    if width_height_type == 'real-time':
        NPIX = row['NPIX']
        k = row['NAXIS1']/row['NAXIS2']
        width = math.sqrt(NPIX * k / math.pi)
        height = math.sqrt(NPIX  / math.pi * k)
        return width,height
    if width_height_type == 'bounding-box':
        width = row['NAXIS1']
        height = row['NAXIS2']
        return width,height
    if width_height_type == 'max-size':
        radius_row = max_radius_rows.loc[max_radius_rows['HARPNUM'] == row['HARPNUM']]
        width = int(radius_row.iloc[0]['NAXIS1'])
        height = int(radius_row.iloc[0]['NAXIS2'])
        return width,height
    raise ValueError('You have selected an invalid dimension type: {}. Please select: real-time,bounding-box,or max-size'.format(width_height_type))

def get_bitmap(fits_file_path):#run ignore warnings to kill warnings from the second line of this function 
    bitmap = sunpy.map.Map(fits_file_path)
    bitmap = bitmap.data
    bitmap[np.add(bitmap == 1,bitmap == 2)] = 0#set off_harp values
    off_harp = (bitmap == 0)
    on_harp = np.logical_not(off_harp)
    bitmap[on_harp] = 1#everything else set to 1
    bitmap = np.rot90(bitmap,2)#bitmaps are taken at a 180 degree angle
    return bitmap

def edge_check(xc,yc,width,height):
    if (xc < 0):
        xc = 0
    if (yc < 0):
        yc = 0
    if (xc + width > 4095):
        xc = 4095 - width
    if (yc + height > 4095):
        yc = 4095 - height
    return xc,yc

def plot_bitmap(xc,yc,bitmap,layer):
    xc = int(xc)
    yc = int(yc)
    layer[yc-bitmap.shape[0]:yc,xc:xc+bitmap.shape[1]] = bitmap
    return layer

def generate_cropped_stack(image_list,size):
    return np.stack([np.array(disk.crop((CROP_FACTOR * size,CROP_FACTOR * size,size-(CROP_FACTOR * size),size-(CROP_FACTOR * size))).resize((size,size), Image.BICUBIC)) for disk in image_list],axis=0)

def save_stack(image_list,current_time,size,dir_name,cropped):
    data_dir = dir_name + '_' + str(size)
    if not cropped:
        savedir = os.path.join(data_dir,current_time.strftime('%Y%m%d_%H%M%S') + '_' + str(size))
        np.save(savedir, np.stack([np.array(disk) for disk in image_list],axis=0))#os.path.join
    else:
        savedir = os.path.join(data_dir,current_time.strftime('%Y%m%d_%H%M%S') + '_' + str(size))
        np.save(savedir, generate_cropped_stack(image_list,size))

def set_to_binary(array):
    array_avg = np.mean(array)
    array[array > array_avg] = 1
    array[array <= array_avg] = 0
    return array

def generate_array_list(image_list,size):
    return [np.array(Image.fromarray(layer).resize((size,size),Image.LANCZOS)) for layer in image_list]

def generate_cropped_array_list(image_list,size):
    return [np.array(Image.fromarray(layer).crop((CROP_INDEX,CROP_INDEX,4096-CROP_INDEX,4096-CROP_INDEX)).resize((size,size),Image.LANCZOS)) for layer in image_list]

def save_bitmap_stack(image_list,current_time,size,dir_name,cropped):#refactor
    data_dir = dir_name + '_' + str(size)
    if not cropped:
        array_list = generate_array_list(image_list,size)
        savedir = os.path.join(data_dir,current_time.strftime('%Y%m%d_%H%M%S') + '_' + str(size))
        np.save(savedir, np.stack([set_to_binary(array) for array in array_list],axis=0))
    else:
        array_list = generate_cropped_array_list(image_list,size)
        savedir = os.path.join(data_dir,current_time.strftime('%Y%m%d_%H%M%S') + '_' + str(size))
        np.save(savedir, np.stack([set_to_binary(array) for array in array_list],axis=0))
    
def generate_filled_circle(start,end,sizes,dir_name,cropped = False,dim_type = 'real-time'):
    generate_sizedirs(sizes,dir_name)
    number_disks = int((end-start)/TIME_INTERVAL + 1)
    print('Generating filled_circles: cropped = {} dim_type = {}, {} thru {} at sizes {}'.format(cropped,dim_type,start,end,sizes))
    for i in range(number_disks):
        current_time = start + i*TIME_INTERVAL
        timestring = current_time.strftime('%Y.%m.%d_%X_TAI')
        rows = df.loc[df['T_REC'] == timestring].reset_index()
        if (not(rows.empty)):
            for size in sizes:
                image_list = []
                for index, row in rows.iterrows():#iterate over rows
                    xc,yc = get_coords(row,current_time)
                    radius = get_radius(row,dim_type)
                    xc,yc,radius = np.array([xc,yc,radius]) * size / 4096#convert 4096-sized info to current disk size
                    disk = Image.new('1', (size, size), color='black') # black/white canvas, initialized to zeros
                    draw = ImageDraw.Draw(disk)
                    draw.ellipse((xc-radius,yc-radius,xc+radius,yc+radius), 'white')
                    image_list.append(disk)
                save_stack(image_list,current_time,size,dir_name,cropped)
                
def generate_filled_ellipse(start,end,sizes,dir_name,cropped = False,dim_type = 'real-time'):
    generate_sizedirs(sizes,dir_name)
    number_disks = int((end-start)/TIME_INTERVAL + 1)
    print('Generating filled_ellipses: cropped = {} dim_type = {}, {} thru {} at sizes {}'.format(cropped,dim_type,start,end,sizes))
    for i in range(number_disks):
        current_time = start + i*TIME_INTERVAL
        timestring = current_time.strftime('%Y.%m.%d_%X_TAI')
        rows = df.loc[df['T_REC'] == timestring].reset_index()
        if (not(rows.empty)):
            for size in sizes:
                image_list = []
                for index, row in rows.iterrows():#iterate over rows
                    xc,yc = get_coords(row,current_time)
                    width,height = get_width_height(row,dim_type)
                    xc,yc,width,height = np.array([xc,yc,width,height]) * size / 4096#convert 4096-sized info to current disk size
                    disk = Image.new('1', (size, size), color='black') # black/white canvas, initialized to zeros
                    draw = ImageDraw.Draw(disk)
                    draw.ellipse((xc-width,yc-height,xc+width,yc+height), 'white')
                    image_list.append(disk)
                save_stack(image_list,current_time,size,dir_name,cropped)

def generate_gaussian_circle(start,end,sizes,dir_name,cropped = False,dim_type = 'real-time'):
    generate_sizedirs(sizes,dir_name)
    number_disks = int((end-start)/TIME_INTERVAL + 1)
    print('Generating gaussian_circles: cropped = {} dim_type = {}, {} thru {} at sizes {}'.format(cropped,dim_type,start,end,sizes))
    for i in range(number_disks):
        current_time = start + i*TIME_INTERVAL
        timestring = current_time.strftime('%Y.%m.%d_%X_TAI')
        rows = df.loc[df['T_REC'] == timestring].reset_index()
        if (not(rows.empty)):
            for size in sizes:
                image_list = []
                for index, row in rows.iterrows():#iterate over rows
                    xc,yc = get_coords(row,current_time)
                    radius = get_radius(row,dim_type)
                    xc,yc,radius = np.array([xc,yc,radius]) * size / 4096#convert 4096-sized info to current disk size
                    x,y = np.meshgrid(range(size),range(size))
                    disk = np.exp(-.5 * SD_FACTOR**2 * (np.square(x-xc) + np.square(y-yc)) / radius**2)
                    disk[disk < THRESHOLD] = 0
                    image_list.append(disk)
                save_stack([Image.fromarray(layer) for layer in image_list],current_time,size,dir_name,cropped)        
    
def generate_gaussian_ellipse(start,end,sizes,dir_name,cropped = False,dim_type = 'real-time'):
    generate_sizedirs(sizes,dir_name)
    number_disks = int((end-start)/TIME_INTERVAL + 1)
    print('Generating gaussian_ellipses: cropped = {} dim_type = {}, {} thru {} at sizes {}'.format(cropped,dim_type,start,end,sizes))
    for i in range(number_disks):
        current_time = start + i*TIME_INTERVAL
        timestring = current_time.strftime('%Y.%m.%d_%X_TAI')
        rows = df.loc[df['T_REC'] == timestring].reset_index()
        if (not(rows.empty)):
            for size in sizes:
                image_list = []
                for index, row in rows.iterrows():#iterate over rows
                    xc,yc = get_coords(row,current_time)
                    width,height = get_width_height(row,dim_type)
                    xc,yc,width,height = np.array([xc,yc,width,height]) * size / 4096#convert 4096-sized info to current disk size
                    k = width/height
                    x,y = np.meshgrid(range(size),range(size))
                    disk = np.exp(-.5 * SD_FACTOR**2 * (np.square(x-xc) + np.square(k * (y-yc))) / width**2)
                    disk[disk < THRESHOLD] = 0
                    image_list.append(disk)
                save_stack([Image.fromarray(layer) for layer in image_list],current_time,size,dir_name,cropped)

def generate_filled_bitmap(start,end,sizes,dir_name,cropped = False):
    generate_sizedirs(sizes,dir_name)
    number_disks = int((end-start)/TIME_INTERVAL + 1)
    print('Generating filled_bitmaps: cropped = {}, {} thru {} at sizes {}'.format(cropped,start,end,sizes))
    warnings.simplefilter("ignore")
    for i in range(number_disks):
        current_time = start + i*TIME_INTERVAL
        timestring = current_time.strftime('%Y.%m.%d_%X_TAI')
        rows = df.loc[df['T_REC'] == timestring].reset_index()
        if (not(rows.empty)):
            image_list = []
            for index, row in rows.iterrows():#iterate over rows
                bitmap_dir = row['FILEDIR']
                bitmap = get_bitmap(bitmap_dir)
                xc = row['CRPIX1'] + row['IMCRPIX1']
                yc = row['CRPIX2'] + row['IMCRPIX2']
                width,height = bitmap.shape
                xc,yc = edge_check(xc,yc,width,height)
                layer = np.zeros((4096,4096))
                layer = plot_bitmap(xc,yc,bitmap,layer)
                image_list.append(layer)
            for size in sizes:
                save_bitmap_stack(image_list,current_time,size,dir_name,cropped)
                
#check savedur 
savedir = '/nobackup/afeghhi/HMI_Data'
#savedir = 'C:/Users/alexf/Desktop/HMI_Data'
verify_dir(savedir)
Ydata_dir =  os.path.join(savedir, 'Ydata')
verify_dir(Ydata_dir)
#load dataframe. Set maxradiusrows and the radius dictionary
df = load_df((os.path.join(savedir,'HARP_df.pickle')))#dataframe must be loaded before generating any disks
max_radius_rows = generate_max_radius_rows(df)#run if generating ellipses at max size
radius_dict = generate_radius_dict(max_radius_rows)#run if generating circles at max size
#set start time, end time, and sizes to save in
start = datetime(2010, 5, 1,0,0,0)#date time object format is year, month, day, hour, minute, second
end = datetime(2011,5, 1, 0,0,0)#the end time is included amongst disks generated change back to year later
sizes = [256,512]
#delete and regenerate Ydirectory
shutil.rmtree(Ydata_dir)
os.mkdir(Ydata_dir)
#run commands to generate data
generate_filled_bitmap(start,end,sizes,os.path.join(Ydata_dir,'filled_bitmaps_uncropped'),cropped = False)
os.system('chmod -R +777 ' + Ydata_dir)