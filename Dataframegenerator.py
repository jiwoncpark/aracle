import os
import astropy.units as u
import warnings
import pickle
import pandas as pd
from sunpy.net import Fido,attrs
from datetime import date, time, datetime, timedelta
from astropy.io import fits

TIME_INTERVAL = timedelta(minutes = 60)
DOWNLOAD_CHUNK = timedelta(days = 10)#avoid download chunks greater than 1 month in order to not download too much at once
EMAIL = 'hsmgroupnasa@gmail.com'
DATA_SERIES = 'hmi.Sharp_720s'
DATA_SEGMENT = 'bitmap'
KEYWORDS = ['HARPNUM','T_REC','NAXIS1','NAXIS2','CDELT1','CDELT2','CRPIX1','CRPIX2','IMCRPIX1','IMCRPIX2','LAT_FWT','LON_FWT','NPIX']#Keywords in order to be saved

def verify_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory {} does not exist. Creating...".format(dir_name))
            
def download_single_chunk(start,end,download_dir):
    response = Fido.search(
    attrs.jsoc.Time(start, end),
    attrs.jsoc.Notify(EMAIL),
    attrs.jsoc.Series(DATA_SERIES),
    attrs.jsoc.Segment(DATA_SEGMENT),
    attrs.Sample(TIME_INTERVAL.total_seconds() * u.s)
    )
    Fido.fetch(response, path= sharp_dir + '/{file}.fits')

def download(start,end,download_dir):
    current_time = start
    while(current_time < end):
        if(end-current_time > DOWNLOAD_CHUNK):
            next_time = current_time + DOWNLOAD_CHUNK
            download_single_chunk(current_time,next_time,download_dir)
        else:
            next_time = end
            download_single_chunk(current_time,next_time,download_dir)
        current_time = current_time + DOWNLOAD_CHUNK

def extract_keywords(fits_file_path,keep_filedir):
    hdul = fits.open(fits_file_path)
    hdul.verify('fix')
    line = []
    for keyword in KEYWORDS:
        line.append(hdul[1].header[keyword])
    if keep_filedir:
        line.append(fits_file_path)
    return line

def generate_df(df_path,sharp_dir,keep_filedir = True):
    if os.path.exists(df_path):
        df = pickle.load(open(df_path,'rb'))
        fits_filenames = os.listdir(sharp_dir)
        fits_filenames = [fits_filename for fits_filename in fits_filenames if not os.path.join(sharp_dir,fits_filename) in df['FILEDIR']]
        if fits_filenames:
            rows = []
            warnings.simplefilter("ignore")#.verify('fix') produces many warnings which will lag the jupyter notebook
            for fits_file_path in fits_filenames:
                rows.append(extract_keywords(os.path.join(sharp_dir,fits_file_path),keep_filedir))
            df.append(rows)
        pickle.dump(df,open(df_path,'wb'))
    else:
        fits_filenames = os.listdir(sharp_dir)
        rows = []
        header = KEYWORDS.copy()
        if keep_filedir:
            header.append('FILEDIR')
        warnings.simplefilter("ignore")#.verify('fix') produces many warnings which will lag the jupyter notebook
        for fits_file_path in fits_filenames:
            rows.append(extract_keywords(os.path.join(sharp_dir,fits_file_path),keep_filedir))
        df = pd.DataFrame(rows,columns = header)
        pickle.dump(df,open(df_path,'wb'))

def del_df(df_path):
    os.remove(df_path)
        
workdir = 'C:/Users/alexf/Desktop/HMI_Data/'
verify_dir(workdir)
sharp_dir = os.path.join(workdir,'sharp')
verify_dir(sharp_dir)
df_path = os.path.join(workdir,'HARP_df.pickle')

#Define start and times, as well as the time interval between disks and the download chunk size
start = datetime(2010, 10, 1,0,0,0)#date time object format is year, month, day, hour, minute, second
end = datetime(2010, 11, 1,0,0,0)#currently generating 8 years of data

#download(start,end,sharp_dir)
#del_df(df_path)
generate_df(df_path,sharp_dir)