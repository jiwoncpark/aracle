import drms #pip install drms, astropy, sunpy , skvideo
import numpy as np
import astropy.units as u
import shutil
import os
import datetime
import matplotlib.pyplot as plt
import skvideo.io
from astropy.io import fits
from matplotlib.pyplot import imshow
from PIL import Image
from sunpy.map import Map
from datetime import date, time, datetime, timedelta
workdir = 'C:/Users/alexf/Desktop/HMI_Data/'
fits_dir = workdir + 'fits/'
if not os.path.exists(workdir):
    os.mkdir(workdir)
    print("Directory " + workdir + "does not exist. Creating...")

start = datetime(2010,5,1,1,0,0)#date time object format is year, month, day, hour, minute, second
end = datetime(2018,5,1,0,0,0)
time_interval = timedelta(minutes = 60) #timedelta will accept weeks,days,hours,minutes and seconds as input
chunk_size = 480 #chunk size is the number of hmi files downloaded in each export call. must be at least 1
export_protocol = 'fits'#using as-is instead of fits will result in important metadata not being downloaded
email = 'hsmgroupnasa@gmail.com'#use a group email
series = 'hmi.M_720s'

if (end < start):
    print("The end date is before the start date. Please select an end date after the start date")
    #sys.exit()
if not os.path.exists(fits_dir):
    os.mkdir(fits_dir)
    print("Directory " + fits_dir + "does not exist. Creating...")
    
c = drms.Client(email=email, verbose = True) 
total = (end-start) // time_interval + 1
print('Downloading ' + str(total) + ' files')
missing_files = []
def download(start,end,chunk_size,time_interval):
    current_time = start
    while(current_time<end):
        if (end-current_time > (time_interval * chunk_size)):
            time_chunk = (time_interval * chunk_size)
        else:
            time_chunk = end-current_time
        end_time = current_time + time_chunk
        current_timestring = current_time.strftime('%Y' + '.' + '%m' + '.'+'%d'+'_'+'%X') + '_UT'
        end_timestring = end_time.strftime('%Y' + '.' + '%m' + '.'+'%d'+'_'+'%X') + '_UT'
        query = series + '[' + current_timestring + '-' + end_timestring + '@' + str(time_interval.total_seconds()) + 's]'
        print('Query string: ' + query)
        try:
            r = c.export(query, protocol = export_protocol)
            r.download(fits_dir)
            exists = os.path.isfile(fits_dir + '.1')
            if exists:#if a fits file no longer exists, it will be downloaded as an empty .1 file. this deletes .1 files
                os.remove(fits_dir + '.1')
                raise ValueError('Fits file no longer exists. Deleting downloaded file...')
        except:#if files are missing from the server, the export call fails. this keeps track of missing files
            if (chunk_size == 1):
                missing_files.append(current_timestring)
            else:
                download(current_time,end_time,chunk_size//2,time_interval)
        current_time = end_time
        
download(start,end,chunk_size,time_interval)
print(missing_files)
#delete all duplicate files
test = os.listdir(fits_dir)

for item in test:
    if item.endswith(".1"):
        os.remove(os.path.join(fits_dir, item))

Xdata_dir =  workdir + 'Xdata/'

if not os.path.exists(Xdata_dir):
    os.mkdir(Xdata_dir)
    print("Directory " + Xdata_dir + "does not exist. Creating...")

fits_filenames = os.listdir(fits_dir)
resizing = [256]
for resize in resizing:
    resize_dir = Xdata_dir + str(resize)
    if  os.path.exists(resize_dir):#delete any resizing directories matching the new resizes
        shutil.rmtree(resize_dir)
    os.makedirs(resize_dir)#creates new resize directories
for filename in fits_filenames: #iterates over fits files and converts to a numpy array
    hmi_map = Map(fits_dir + filename)
    rotateddata90 = hmi_map.rotate(angle=90*u.deg, order = 0)
    rotateddata180 = rotateddata90.rotate(angle=90*u.deg, order = 0)
    data = rotateddata180.data
    data[np.where(np.isnan(data))] = 0.0      # replacing nans with 0s
    print('saving '+filename +' in sizes'+ str(resizing))
    for resize in resizing:#resizes and saves numpy array data into given resizes
        resized_image = np.array(Image.fromarray(data).resize((resize,resize),Image.LANCZOS))
        np.save(Xdata_dir + str(resize) + '/' + filename[:26] + '_'+ str(resize), resized_image)#saves series,time,and resize