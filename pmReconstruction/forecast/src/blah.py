# install netCDF4 using: conda install -c conda-forge netcdf4
# install ECMWF API: pip install ecmwf-opendata==0.0.14
# conda activate pmtest

import s3fs
from ecmwf.opendata import Client
import os
from os.path import exists
#from datetime import datetime
import datetime
import pandas as pd
import pygrib
import xarray as xr
from pyproj import Proj
from pysolar.solar import *
import sys
#import cProfile
import time as ct
from scipy.spatial.distance import cdist
import numpy as np
from scipy.interpolate import griddata



start_time = ct.time()

save_dir = '/scratch/prabuddha/pm_est_fc/data/raw_data/'
# where the ancillary dataFrame saved
ancillary_df = '/scratch/prabuddha/pm_est/ref_data/USA_latlon_ancillary_df_final.csv'
save_raw_df = '/scratch/prabuddha/pm_est_fc/data/raw_df/'


fs = s3fs.S3FileSystem(anon=True)
client = Client(source="ecmwf")

# crnt_time is current UTC time
crnt_time = datetime.datetime.now() + datetime.timedelta(hours=5)
str_crnt_time = str(crnt_time)
year, month, day, hour = str_crnt_time[0:4], str_crnt_time[5:7], str_crnt_time[8:10], str_crnt_time[11:13]
print("Current UTC time " + str_crnt_time)


# meteoro data has two key words - time and step
# time is when the forecast start (normally forecast start 10 or 12 hrs behind the current UTC).
# time keyword can have only 00, 06, 12, 18.
# to get the current UTC data, we have to find the closest value for time by substracting 10 hrs from current UTC.
# Then add multiplication of 3 until close to the current UTC.
# this multiplication value is the step.

if 0 <= int(hour) < 6:
    time = "12"
    if int(hour) < 3:
        step = 12
        new_hour = "00"
    else:
        step = 15
        new_hour = "03"

elif 6 <= int(hour) < 12:
    time = "18"
    if int(hour) < 9:
        step = 12
        new_hour = "06"
    else:
        step = 15
        new_hour = "09"

elif 12 <= int(hour) < 18:
    time = "00"
    if int(hour) < 15:
        step = 12
        new_hour = "12"
    else:
        step = 15
        new_hour = "15"

else:
    time = "06"
    if int(hour) < 21:
        step = 12
        new_hour = "18"
    else:
        step = 15
        new_hour = "21"

print("downloading data for " + year + "-" + month + "-" + day + " " + new_hour + ":00")


try:
    os.mkdir(save_dir + year)
except FileExistsError:
    print("year folder exists")

try:
    os.mkdir(save_dir + year + '/' + month)
except FileExistsError:
    print("month folder exists")

try:
    os.mkdir(save_dir + year + '/' + month + '/' + day)
except FileExistsError:
    print("day folder exists")

try:
    os.mkdir(save_dir + year + '/' + month + '/' + day + '/' + new_hour)
except FileExistsError:
    print("hour folder exists")


meteo_dir = save_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + "meteo_" + new_hour + '.grib2'
AOD_dir = save_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + "AOD_" + new_hour


def down_meteo():
    result = client.retrieve(
        time=time,
        stream = "oper",
        type = "fc",
        #step=[3,6],
        step = step,
        param = ["2t", "r", "sp", "10u", "10v", "tp", "q"],
        #param = ["tp"],
        target = meteo_dir,
    )
    print(result.datetime)


def down_AOD():
    date_time = year + "-" + month + "-" + day + " " + new_hour + ":00:00"
    datetime_object = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    julian_day = datetime_object.strftime('%j')
    digit = len(str(julian_day))
    if digit == 1:
        julian_day = '0' + '0' + str(julian_day)
    elif digit == 2:
        julian_day = '0' + str(julian_day)
    else:
        julian_day = str(julian_day)
    AWS_path = 'noaa-goes16/ABI-L2-AODC/' + str(year) + '/' + julian_day + '/' + new_hour + '/*'
    files = fs.glob(AWS_path)
    fs.get(files[0], AOD_dir)

stage1_time = ct.time()
execution1_time = stage1_time - start_time
print(f"Setting up time: {execution1_time} seconds")


if exists(meteo_dir):
    print("meteorogical data exist")
else:
    down_meteo()

if exists(AOD_dir):
    print("AOD data exist")
else:
    down_AOD()

stage2_time = ct.time()
execution2_time = stage2_time - stage1_time
print(f"Downloading time: {execution2_time} seconds")

#import time
#ct.sleep(30)

###############################################################
# Make a DataFrame
coord_df = pd.read_csv(ancillary_df)
#print(len(coord_df))
#coord = coord_df[0:10]
coord = coord_df
#print(coord)


stage3_time = ct.time()
execution3_time = stage3_time - stage2_time
print(f"Variable assign time: {execution3_time} seconds")





######### matching meteorological data #############################
meteo_grbs = pygrib.open(meteo_dir)
detail = meteo_grbs.select()
grb_temp = meteo_grbs.select(name='2 metre temperature')[0]
mete_lats, mete_lons = grb_temp.latlons()

# Interpolate the selected variables onto the latitude and longitude coordinates in the dataframe
selected_vars = ['2 metre temperature', 'Relative humidity', 'Surface pressure', '10 metre U wind component', '10 metre V wind component', 'Specific humidity']
interpolated_data = dict((var, griddata((mete_lats.ravel(), mete_lons.ravel()), meteo_grbs.select(name=var)[0].values.ravel(), (coord['latitude'], coord['longitude']), method='linear')) for var in selected_vars)

coord.reset_index(drop=True, inplace=True)
# Add the interpolated variable values to the dataframe
coord = pd.concat([coord, pd.DataFrame(interpolated_data)], axis=1)

# total precipitation data not always available
try:
    preci = meteo_grbs.select(name='Total Precipitation')[0]
    preci_val = preci.values
    interpolated_preci_data = griddata((mete_lats.ravel(), mete_lons.ravel()), preci_val.ravel(), (coord['latitude'], coord['longitude']), method='linear')
    coord['total_preci'] = interpolated_preci_data
except:
    print('no precipitation data')
    coord['precipitation'] = 0




################ Matching AOD data ###############################

AOD_data = xr.open_dataset(AOD_dir, engine='netcdf4')
sat_sweep =  AOD_data.variables['goes_imager_projection'].attrs["sweep_angle_axis"]
sat_h = AOD_data.variables['goes_imager_projection'].attrs["perspective_point_height"]
xrcor = AOD_data.assign_coords(x=(AOD_data.x)*sat_h,y=(AOD_data.y)*sat_h)
#pro = Proj("+proj=geos +lon_0=-75 +h=" +str(round(sat_h)) + " +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True")
pro = Proj("+proj=geos +lon_0=-75 +h=35786023 + x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True") 


AOD_val = AOD_data.AOD.data
DQF_val = AOD_data.DQF.data
xx_goes, yy_goes = np.meshgrid(AOD_data.x.data*sat_h, AOD_data.y.data*sat_h)
aod_lons, aod_lats = pro(xx_goes, yy_goes, inverse=True)
# removing inf from the coordinates
lon_inf_mask = np.isinf(aod_lons)
aod_lons[lon_inf_mask] = np.nan
lat_inf_mask = np.isinf(aod_lats)
aod_lats[lat_inf_mask] = np.nan
valid_indices = ~np.isnan(aod_lons) & ~np.isnan(aod_lats)

# remove NaN values and their corresponding t values
aod_lon_clean = aod_lons[valid_indices]
aod_lat_clean = aod_lats[valid_indices]
AOD_val_clean = AOD_val[valid_indices]
DQF_val_clean = DQF_val[valid_indices]

xx = coord['longitude']
yy = coord['latitude']

AOD_interp = griddata((aod_lon_clean, aod_lat_clean), AOD_val_clean, (xx, yy), method='linear')
DQF_interp = griddata((aod_lon_clean, aod_lat_clean), DQF_val_clean, (xx, yy), method='nearest')
coord['AOD'] = AOD_interp
coord['DQF'] = DQF_interp





################ calculate the solar azimuthal and zenith angles for each row in the DataFrame
SA_date = datetime.datetime(int(year), int(month), int(day), int(new_hour), tzinfo=datetime.timezone.utc)
coord['SZA'] = [float(90) - get_altitude(lat, lon, SA_date) for lat, lon in zip(coord['latitude'], coord['longitude'])]
coord['SAA'] = [get_azimuth(lat, lon, SA_date) for lat, lon in zip(coord['latitude'], coord['longitude'])]
#coord = pd.merge(coord, aod_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])
#print(coord)
coord = coord.rename(columns={'2 metre temperature':'temperature','Surface pressure':'pressure','10 metre U wind component':'u_wind','10 metre V wind component':'v_wind',
    'Relative humidity':'rel_humidity', 'Specific humidity':'spec_humidity'})
  
    



################ Forecast PBLH data download and match ###############################################################################################################################
import datetime as dt
from datetime import datetime

now = datetime.now()
now_hour = now.hour
values = [0, 6, 12, 18]
if now_hour <= values[0]:
    pblh_hour_srt = str(values[0])
else:
    closest = max(filter(lambda x: x <= now_hour, values))
    pblh_hour_srt = str(closest)
    
# fill zeros if the number length is less than 2
pblh_hour_srt = str(pblh_hour_srt).zfill(2)
lowest_datetime_str = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + 'T' + pblh_hour_srt + ':00:00'
lowest_datetime = datetime.strptime(lowest_datetime_str, "%Y-%m-%dT%H:%M:%S") - dt.timedelta(hours=6)
date_time = str(lowest_datetime.year) + str(lowest_datetime.month).zfill(2) + str(lowest_datetime.day).zfill(2) + '_' + str(lowest_datetime.hour).zfill(2)


url = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/tavg1_2d_flx_Nx/tavg1_2d_flx_Nx." + date_time

#url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/tavg1_2d_flx_Nx/tavg1_2d_flx_Nx.' + str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2) + '_' + pblh_hour_srt
ds = xr. open_dataset(url)

pblh_srt = year + '-' + month + '-' + day + 'T' + new_hour + ':00:00'
pblh_end = datetime.strptime(pblh_srt, "%Y-%m-%dT%H:%M:%S") + dt.timedelta(hours=1)
pblh_end_str = str(pblh_end.year) + '-' + str(pblh_end.month).zfill(2) + '-' + str(pblh_end.day).zfill(2) + 'T' + str(pblh_end.hour).zfill(2) + ':00:00'

time_range = slice(pblh_srt, pblh_end_str)
lat_range = slice(25, 50)
lon_range = slice(-125, -66)
ds_1hr = ds.sel(time=time_range)
ds_1hr = ds_1hr.sel(lat=lat_range)
ds_1hr = ds_1hr.sel(lon=lon_range)

pblh_time = ds_1hr.time.values
pblh_lat = ds_1hr.lat.values
pblh_lon = ds_1hr.lon.values
pblh = ds_1hr.pblh.values[0]

co_xx = coord['longitude']
co_yy = coord['latitude']

xx_pblh, yy_pblh = np.meshgrid(pblh_lon, pblh_lat)
x = xx_pblh.ravel()
y = yy_pblh.ravel()
z = pblh.ravel()
pblh_interp = griddata((x, y), z, (co_xx, co_yy), method='linear')
coord['pblh'] = pblh_interp
print(coord)
#coord.to_csv('intrp_pblh_df.csv')



down_dir = save_raw_df
try:
    os.mkdir(down_dir + year)
except FileExistsError:
    print("year folder exists")
try:
    os.mkdir(down_dir + year + '/' + month)
except FileExistsError:
    print("month folder exists")
try:
    os.mkdir(down_dir + year + '/' + month + '/' + day)
except FileExistsError:
    print("day folder exists")
try:
    os.mkdir(down_dir + year + '/' + month + '/' + day + '/' + new_hour)
except FileExistsError:
    print("hour folder exists")
#out_df_path = down_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + 'df_part_test_a_' + str(ind) + '.csv'
out_df_path = down_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + 'df_' + year + '_' + month + '_' + day + '_' + new_hour + '.csv'
coord.to_csv(out_df_path)


stage4_time = ct.time()
execution4_time = stage4_time - stage3_time
print(f"Matching time: {execution4_time} seconds")


