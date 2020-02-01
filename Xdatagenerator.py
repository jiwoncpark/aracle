import drms #pip install drms, astropy, sunpy , skvideo
import numpy as np
import shutil
import os
from astropy.io import fits
from PIL import Image
from sunpy.map import Map
from datetime import date, time, datetime, timedelta

TIME_INTERVAL = timedelta(minutes = 60) #timedelta will accept weeks,days,hours,minutes and seconds as input
CHUNK_SIZE = 480 #chunk size is the number of hmi files downloaded in each export call. must be at least 1
EXPORT_PROTOCOL = 'fits'#using as-is instead of fits will result in important metadata not being downloaded
EMAIL = 'hsmgroupnasa@gmail.com'#use a group email
X_DATA_SERIES = 'hmi.M_720s'
DATA_EXTENSION = '_TAI.1.magnetogram.fits'
C = drms.Client(email=EMAIL, verbose = True) 
CROP_FACTOR = .29289321881 #CROP_FACTOR * 2 is the portion you crop
CROP_INDEX = 4096 * CROP_FACTOR
MAGNETOGRAM_RESIZE = Image.LANCZOS

def verify_dir(dir_name):
    if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Directory {} does not exist. Creating...".format(dir_name))

def generate_sizedirs(sizes,dir_name):
    for size in sizes:
        data_dir = dir_name + '_' + str(size)
        verify_dir(data_dir)
        
def generate_daterange(start,end):
    list_length = int(((end - start) / TIME_INTERVAL) + 1)
    return [start + (TIME_INTERVAL * i) for i in range(list_length)]

def generate_dates(date_intervals_list):
    date_list = []
    for start,end in date_intervals_list:
        date_list.append(generate_daterange(start,end))
    date_list = [item for sublist in date_list for item in sublist]
    return date_list
    
def recursive_download(start,end,current_chunk_size,download_dir):
    current_time = start
    while(current_time<end):
        if (end-current_time > (TIME_INTERVAL * current_chunk_size)):
            time_chunk = (TIME_INTERVAL * current_chunk_size)  - TIME_INTERVAL
        else:
            time_chunk = end-current_time
        next_time = current_time + time_chunk
        current_timestring = current_time.strftime('%Y.%m.%d_%X') + '_UT'
        next_timestring = next_time.strftime('%Y.%m.%d_%X') + '_UT'
        query = '{}[{}-{}@{}s]'.format(X_DATA_SERIES,current_timestring,next_timestring,TIME_INTERVAL.total_seconds())
        try:
            r = C.export(query, protocol = EXPORT_PROTOCOL)
            r.download(download_dir)
        except:
            if (current_chunk_size != 1):
                recursive_download(current_time,next_time,current_chunk_size//2,download_dir)
        current_time = next_time + TIME_INTERVAL
                        
def download_single_chunk(start,end,download_dir):
    start_timestring = start.strftime('%Y.%m.%d_%X') + '_UT'
    end_timestring = end.strftime('%Y.%m.%d_%X') + '_UT'
    query = '{}[{}-{}@{}s]'.format(X_DATA_SERIES,start_timestring,end_timestring,TIME_INTERVAL.total_seconds())
    try:
        r = C.export(query,protocol = EXPORT_PROTOCOL)
        r.download(download_dir)
    except:
        recursive_download(start,end,CHUNK_SIZE,download_dir)
        
def download_fits_files(start,end,download_dir):
    current_time = start
    download_chunk = TIME_INTERVAL * CHUNK_SIZE
    while(current_time < end):
        if(end-current_time > download_chunk):
            next_time = (current_time + download_chunk) - TIME_INTERVAL
            download_single_chunk(current_time,next_time - TIME_INTERVAL,download_dir)
        else:
            next_time = end
            download_single_chunk(current_time,next_time,download_dir)
        current_time = current_time + download_chunk
        
def save_hmi_array(hmi_data,timestamp,size,output_dir,cropped):
    filename = timestamp.strftime('%Y%m%d_%H%M%S') + '_' + str(size)
    savedir = os.path.join(output_dir + '_' + str(size),filename)
    if not cropped:
        resized_hmi_image = np.array(Image.fromarray(hmi_data).resize((size,size),MAGNETOGRAM_RESIZE))
    else:
        hmi_image_cropped = Image.fromarray(hmi_data).crop((CROP_INDEX,CROP_INDEX,4096-CROP_INDEX,4096-CROP_INDEX))
        resized_hmi_image = np.array(Image.fromarray(hmi_image_cropped).resize((size,size),MAGNETOGRAM_RESIZE))
    np.save(savedir,resized_hmi_image)
    
def extract_and_resize(datetime_list,sizes,fits_dir,output_dir,cropped = False):
    generate_sizedirs(sizes,output_dir)
    for current_time in datetime_list:
        filename =  X_DATA_SERIES + '.' + current_time.strftime('%Y%m%d_%H%M%S') + DATA_EXTENSION
        file_dir = os.path.join(fits_dir,filename)
        if os.path.exists(file_dir):
            hmi_map = Map(file_dir)
            hmi_data = hmi_map.data
            hmi_data = np.rot90(hmi_data,2)#rot 180
            hmi_data[np.isnan(hmi_data)] = 0#set nan to 0
            for size in sizes:#resizes and saves numpy array data into given resizes
                save_hmi_array(hmi_data,current_time,size,output_dir,cropped)

#check savedur 
savedir = '/nobackup/afeghhi/HMI_Data'    
#savedir = 'C:/Users/alexf/Desktop/HMI_Data'
verify_dir(savedir)
fits_dir = os.path.join(savedir,'fits')
verify_dir(fits_dir)
Xdata_dir = os.path.join(savedir, 'Xdata')
verify_dir(Xdata_dir)
start = datetime(2010,5,1,0,0,0)#date time object format is year, month, day, hour, minute, second
end = datetime(2011,5,1,0,0,0)
#download_fits_files(start,end,fits_dir)
sizes = [256]
dates = generate_daterange(start,end)
#delete and regenerate Xdirectory
shutil.rmtree(Xdata_dir)
verify_dir(Xdata_dir)
#run code to generate magnetograms
uncropped_dir = os.path.join(Xdata_dir,'uncropped_hmis')
cropped_dir = os.path.join(Xdata_dir,'cropped_hmis')
extract_and_resize(dates,sizes,fits_dir,uncropped_dir,cropped = False)
extract_and_resize(dates,sizes,fits_dir,cropped_dir,cropped = True)
os.system('chmod -R +777 ' + Xdata_dir)