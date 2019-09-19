import numpy as np
import astropy.units as u
import os
import sunpy.coordinates
import shutil
import pandas as pd
from datetime import date, time, datetime, timedelta
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from PIL import Image, ImageDraw

workdir = '/nobackup/afeghhi/HMI_Data/'
Ydata_dir =  workdir + 'Ydata/'
Ydata_cut =  workdir + 'Ydata_cut/'

if not os.path.exists(workdir):
    os.mkdir(workdir)
    print("Directory " + workdir + "does not exist. Creating...")

if not os.path.exists(Ydata_dir):
    os.mkdir(Ydata_dir)
    print("Directory " + Ydata_dir + "does not exist. Creating...")
    
if not os.path.exists(Ydata_cut):
    os.mkdir(Ydata_cut)
    print("Directory " + Ydata_cut + "does not exist. Creating...")

data = open(workdir + 'data.txt','r')
lines = data.readlines()
header = lines[0].split()
del lines[0]
df = pd.DataFrame([line.split() for line in lines],columns=header)

max_radius_rows = df[df.groupby(['HARPNUM'])['NPIX'].transform(max) == df['NPIX']]#isolate all rows with the max npix per harp
max_radius_rows['RADIUS'] = max_radius_rows[['NAXIS1', 'NAXIS2']].max(axis=1)/2#find the max height/width then divide by 2
radius_dict = pd.Series(max_radius_rows.RADIUS.values,index=max_radius_rows.HARPNUM).to_dict()

start = datetime(2010, 5, 1,0,0,0)#date time object format is year, month, day, hour, minute, second
end = datetime(2011,5, 1, 0,0,0)#the end time is included amongst disks generated
time_interval = timedelta(minutes = 60)

output_sizes = [256]
for size in output_sizes:
    resize_dir = Ydata_dir + str(size)
    if  os.path.exists(resize_dir):#delete any resizing directories matching the new resizes
        shutil.rmtree(resize_dir)
    os.makedirs(resize_dir)
    resize_dir_cut = Ydata_cut + str(size)
    if  os.path.exists(resize_dir_cut):#delete any resizing directories matching the new resizes
        shutil.rmtree(resize_dir_cut)
    os.makedirs(resize_dir_cut)

number_disks = int((end-start)/time_interval + 1)
for i in range(number_disks):
    time = start + i*time_interval
    timestring = time.strftime('%Y' + '.' + '%m' + '.' + '%d' + '_' + '%X' + '_TAI')
    obstimestring = time.strftime('%Y' + '-' + '%m' + '-' + '%d')
    rows = df.loc[df['T_REC'] == timestring].reset_index()
    image_list = []
    if (not(rows.empty)):
        for index, row in rows.iterrows():#iterate over rows
            disk = Image.new('1', (4096, 4096), color='black') # black/white canvas, initialized to zeros
            draw = ImageDraw.Draw(disk)
            hpc1 = SkyCoord(float(rows.iloc[index]['LON_FWT'])*u.deg, float(rows.iloc[index]['LAT_FWT'])*u.deg, frame=frames.HeliographicStonyhurst, obstime=obstimestring)
            hpc_out = sunpy.coordinates.Helioprojective(observer='earth', obstime=obstimestring) # arcsec
            hpc2 = hpc1.transform_to(hpc_out) # convert to arcsecond
            xc = (hpc2.Tx / u.arcsec) # stripping units
            yc = (hpc2.Ty / u.arcsec)
            xc = (xc/float(rows.iloc[index]['CDELT1'])) + float(rows.iloc[index]['IMCRPIX1']) # convert to pixel value
            yc = (yc/float(rows.iloc[index]['CDELT2'])) + float(rows.iloc[index]['IMCRPIX2'])
            radius = radius_dict[rows.iloc[index]['HARPNUM']]
            draw.ellipse((xc-radius,yc-radius,xc+radius,yc+radius), 'white')
            image_list.append(np.flipud(np.array(disk.resize((256,256), Image.BILINEAR))).astype(int))
    if(image_list):
        np.save(Ydata_dir + str(size) + '/' + time.strftime('%Y' + '%m' + '%d' + '_' + '%H' +'%M' + '%S') + '_' + str(size), np.stack(image_list,axis=0))
