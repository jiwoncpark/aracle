import os
import astropy.units as u
import warnings
from sunpy.net import Fido,attrs
from datetime import date, time, datetime, timedelta
from astropy.io import fits

workdir = 'C:/Users/alexf/Desktop/HMI_Data/'
sharp_dir = workdir + 'sharp/'

if not os.path.exists(workdir):
    os.mkdir(workdir)
    print("Directory " + workdir + "does not exist. Creating...")

if not os.path.exists(sharp_dir):
    os.mkdir(sharp_dir)
    print("Directory " + sharp_dir + "does not exist. Creating...")
#Define start and times, as well as the time interval between disks and the download chunk size
start = datetime(2010, 5, 1,0,0,0)#date time object format is year, month, day, hour, minute, second
end = datetime(2018, 5, 1,0,0,0)#currently generating 8 years of data
time_interval = timedelta(minutes = 60)
download_chunk = timedelta(days = 10)#avoid download chunks greater than 1 month in order to not download too much at once

#breaks the download into pieces and downloads
current_time = start
while(current_time < end):
    if(end-current_time > download_chunk):
        next_time = current_time + download_chunk
    else:
        next_time = end
    response = Fido.search(
    attrs.jsoc.Time(current_time, next_time),
    attrs.jsoc.Notify('hsmgroupnasa@gmail.com'),
    attrs.jsoc.Series('hmi.Sharp_720s'),
    attrs.jsoc.Segment('bitmap'),
    attrs.Sample(time_interval.total_seconds() * u.s)
    )
    response
    res = Fido.fetch(response, path= sharp_dir + '/{file}.fits')
    current_time = next_time

warnings.simplefilter("ignore")#.verify('fix') produces many warnings which will lag the jupyter notebook

#extracts relevant keywords in the given order
keywords = ['HARPNUM','T_REC','NAXIS1','NAXIS2','CDELT1','CDELT2','IMCRPIX1','IMCRPIX2','LAT_FWT','LON_FWT','NPIX']#Keywords in order to be saved
filenames = os.listdir(sharp_dir)
filename = 'data.txt'

data = open(workdir + filename,"w+")
line = ''
for keyword in keywords:
    line += keyword + ' '
data.write(line + "\n")
for filename in filenames:
    line = ''
    hdul = fits.open(sharp_dir + filename)
    hdul.verify('fix')
    for keyword in keywords:
        line += str(hdul[1].header[keyword]) + ' '
    data.write(line + '\n')
    hdul.close()
data.close()